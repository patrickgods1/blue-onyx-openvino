//! OpenVINO core properties, model loading on one device, tensor IO. The GPU->CPU fallback is
//! the caller's candidate loop (`plan.rs`, `WorkerCtx::load`).
//!
//! Built against the `openvino` 0.11 crate (runtime-linking). Port metadata is read from the
//! (possibly reshaped) `Model` before compilation; tensors are addressed by port index so models
//! whose ports have no tensor names still work.

use super::spec::{Runtime, Target};
use super::{Candidate, CoreOptions, LoadRequest, ModelInfo, OvBackend, OvCompiled};
use crate::model::{ExtraData, ExtraInput, NamedOutput, OutputBuf, PortElem, PortSpec};
use anyhow::{Context, Result};
use openvino::{
    DeviceType, ElementType, Model, Node, PartialShape, PropertyKey, RwPropertyKey, Shape, Tensor,
};
use std::time::Instant;

/// Static input shape used when the image input has dynamic dimensions.
pub(crate) const DEFAULT_IMAGE_SHAPE: [i64; 4] = [1, 3, 640, 640];
/// Static shape for RT-DETR's `orig_target_sizes` when dynamic.
pub(crate) const DEFAULT_TARGET_SIZES_SHAPE: [i64; 2] = [1, 2];

/// Which device a model was requested on and ended up on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceInfo {
    /// Device as configured ("GPU", "GPU.1", "CPU", "auto", "ort:cuda", ...).
    pub requested: String,
    /// Runtime device: "GPU", "GPU.1", "CPU" for OpenVINO.
    pub actual: String,
    /// FULL_DEVICE_NAME, e.g. "Intel(R) UHD Graphics 630 (iGPU)"
    pub full_name: String,
    pub fell_back: bool,
    pub runtime: Runtime,
    /// Canonical spec of the candidate that loaded, e.g. "openvino:gpu", "openvino:cpu".
    pub spec: String,
}

impl DeviceInfo {
    pub fn is_gpu(&self) -> bool {
        match self.runtime {
            Runtime::OpenVino => self.actual.starts_with("GPU"),
            Runtime::Ort => super::spec::parse(&self.spec).is_ok_and(|s| s.is_gpu_like()),
        }
    }
    /// String reported as `executionProvider` in API responses, e.g.
    /// "OpenVINO GPU (Intel(R) UHD Graphics 630, fallback)".
    pub fn execution_provider(&self) -> String {
        let kind = match self.runtime {
            // OpenVINO keeps the pre-6.1 wording: GPU or CPU only.
            Runtime::OpenVino if self.is_gpu() => "GPU",
            Runtime::OpenVino => "CPU",
            Runtime::Ort => super::spec::parse(&self.spec)
                .ok()
                .and_then(|s| s.device().map(|d| d.target.display_name()))
                .unwrap_or(Target::Cpu.display_name()),
        };
        let runtime = self.runtime.display_name();
        let fb = if self.fell_back { ", fallback" } else { "" };
        if self.full_name.is_empty() {
            format!("{runtime} {kind}{fb}")
        } else {
            format!("{runtime} {kind} ({}{fb})", self.full_name)
        }
    }
}

fn has_device(available: &[String], prefix: &str) -> bool {
    available.iter().any(|d| d.starts_with(prefix))
}

/// `Core::set_property`, except on Apple arm64 where openvino-sys calls the variadic C function
/// with the wrong ABI (see `libs::core_set_property_variadic`).
fn core_set_property(
    core: &mut openvino::Core,
    device: &DeviceType,
    key: &RwPropertyKey,
    value: &str,
) -> Result<()> {
    #[cfg(all(target_vendor = "apple", target_arch = "aarch64"))]
    {
        super::libs::core_set_property_variadic(core, device.as_ref(), key.as_ref(), value)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    #[cfg(not(all(target_vendor = "apple", target_arch = "aarch64")))]
    {
        core.set_property(device, key, value)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

fn set_prop(core: &mut openvino::Core, device: &DeviceType, key: RwPropertyKey, value: &str) {
    match core_set_property(core, device, &key, value) {
        Ok(()) => tracing::info!("OpenVINO {device}: {}={value}", key.as_ref()),
        Err(e) => tracing::warn!(
            "OpenVINO {device}: failed to set {}={value}: {e}",
            key.as_ref()
        ),
    }
}

pub fn apply_core_properties(
    core: &mut openvino::Core,
    opts: &CoreOptions,
    available: &[String],
) -> Result<()> {
    tracing::info!("OpenVINO available devices: {available:?}");
    let cache = match &opts.cache_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating OpenVINO cache dir {}", dir.display()))?;
            Some(
                dir.to_str()
                    .with_context(|| format!("cache dir {} is not valid UTF-8", dir.display()))?
                    .to_string(),
            )
        }
        None => None,
    };
    let mut devices = Vec::new();
    if has_device(available, "CPU") {
        devices.push(DeviceType::CPU);
    }
    if has_device(available, "GPU") {
        devices.push(DeviceType::GPU);
    }
    for dev in &devices {
        if let Some(c) = &cache {
            set_prop(core, dev, RwPropertyKey::CacheDir, c);
        }
        set_prop(core, dev, RwPropertyKey::HintPerformanceMode, "LATENCY");
        if *dev == DeviceType::CPU && opts.intra_threads > 0 {
            set_prop(
                core,
                dev,
                RwPropertyKey::InferenceNumThreads,
                &opts.intra_threads.to_string(),
            );
        }
    }
    Ok(())
}

pub fn full_device_name(core: &openvino::Core, device: &str) -> String {
    core.get_property(&DeviceType::from(device), &PropertyKey::DeviceFullName)
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| device.to_string())
}

fn port_elem(e: ElementType) -> PortElem {
    match e {
        ElementType::F32 => PortElem::F32,
        ElementType::F16 => PortElem::F16,
        ElementType::I64 => PortElem::I64,
        ElementType::I32 => PortElem::I32,
        ElementType::U8 => PortElem::U8,
        _ => PortElem::Other,
    }
}

/// Dimensions of a partial shape with -1 for dynamic dims; empty for a dynamic rank.
fn partial_dims(ps: &PartialShape) -> Vec<i64> {
    let rank = ps.get_rank();
    if rank.get_min() != rank.get_max() || rank.get_max() < 0 {
        return Vec::new();
    }
    ps.get_dimensions()
        .iter()
        .map(|d| if d.is_dynamic() { -1 } else { d.get_min() })
        .collect()
}

fn port_spec(node: &Node, idx: usize, kind: &str) -> Result<PortSpec> {
    // `get_name` fails for ports without tensor names; synthesize one (IO is index-based).
    let name = node.get_name().unwrap_or_else(|_| format!("{kind}{idx}"));
    let ps = node
        .get_partial_shape()
        .with_context(|| format!("reading shape of {kind} port {idx} ({name})"))?;
    let elem = node
        .get_element_type()
        .with_context(|| format!("reading element type of {kind} port {idx} ({name})"))?;
    Ok(PortSpec {
        name,
        shape: partial_dims(&ps),
        elem: port_elem(elem),
    })
}

fn introspect(model: &Model) -> Result<(Vec<PortSpec>, Vec<PortSpec>)> {
    let n_in = model.get_inputs_len().context("counting model inputs")?;
    let n_out = model.get_outputs_len().context("counting model outputs")?;
    let mut inputs = Vec::with_capacity(n_in);
    for i in 0..n_in {
        let node = model
            .get_input_by_index(i)
            .with_context(|| format!("getting input port {i}"))?;
        inputs.push(port_spec(&node, i, "input")?);
    }
    let mut outputs = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let node = model
            .get_output_by_index(i)
            .with_context(|| format!("getting output port {i}"))?;
        outputs.push(port_spec(&node, i, "output")?);
    }
    Ok((inputs, outputs))
}

/// Index of the image input: named `images`/`input`, else the first 4-D input.
pub(crate) fn find_image_input(inputs: &[PortSpec]) -> Option<usize> {
    inputs
        .iter()
        .position(|p| p.name == "images")
        .or_else(|| inputs.iter().position(|p| p.name == "input"))
        .or_else(|| inputs.iter().position(|p| p.shape.len() == 4))
}

fn read_model(core: &mut openvino::Core, path: &std::path::Path) -> Result<Model> {
    let p = path
        .to_str()
        .with_context(|| format!("model path {} is not valid UTF-8", path.display()))?;
    match core.read_model_from_file(p, "") {
        Ok(m) => Ok(m),
        Err(e) => {
            let is_xml = path
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("xml"));
            let bin = path.with_extension("bin");
            if is_xml && bin.is_file() {
                tracing::warn!(
                    "reading {} with implicit weights failed ({e}); retrying with {}",
                    path.display(),
                    bin.display()
                );
                let b = bin.to_str().with_context(|| {
                    format!("weights path {} is not valid UTF-8", bin.display())
                })?;
                core.read_model_from_file(p, b)
                    .with_context(|| format!("reading model {} (weights {b})", path.display()))
            } else {
                Err(anyhow::Error::new(e).context(format!("reading model {}", path.display())))
            }
        }
    }
}

/// Index of the extra (non-image) input `name`. IR converted from ONNX can keep the ONNX name
/// only as a tensor alias (e.g. RT-DETR's `orig_target_sizes` shows up as
/// `/postprocessor/Expand_output_0`), so fall back to the sole other input when it is 2-D.
pub(crate) fn extra_input_index(
    inputs: &[PortSpec],
    image_idx: usize,
    name: &str,
) -> Option<usize> {
    if let Some(i) = inputs.iter().position(|p| p.name == name) {
        return Some(i);
    }
    let mut others = (0..inputs.len()).filter(|&i| i != image_idx);
    match (others.next(), others.next()) {
        (Some(i), None) if inputs[i].shape.is_empty() || inputs[i].shape.len() == 2 => Some(i),
        _ => None,
    }
}

/// Reshape dynamic image / `orig_target_sizes` inputs to static shapes. Returns true if reshaped.
fn make_static(
    model: &mut Model,
    inputs: &[PortSpec],
    image_idx: usize,
    path: &std::path::Path,
) -> Result<bool> {
    let mut targets: Vec<(usize, Vec<i64>)> = Vec::new();
    let img = &inputs[image_idx];
    if img.shape.len() != 4 && !img.shape.is_empty() {
        anyhow::bail!(
            "model {}: image input '{}' has rank {} (expected 4-D NCHW, shape {:?})",
            path.display(),
            img.name,
            img.shape.len(),
            img.shape
        );
    }
    if img.shape.is_empty() || img.shape.iter().any(|&d| d < 0) {
        let dims: Vec<i64> = (0..4)
            .map(|i| match img.shape.get(i) {
                Some(&d) if d >= 0 => d,
                _ => DEFAULT_IMAGE_SHAPE[i],
            })
            .collect();
        targets.push((image_idx, dims));
    }
    if let Some(i) = extra_input_index(inputs, image_idx, "orig_target_sizes")
        && (inputs[i].shape.is_empty() || inputs[i].shape.iter().any(|&d| d < 0))
    {
        targets.push((i, DEFAULT_TARGET_SIZES_SHAPE.to_vec()));
    }
    if targets.is_empty() {
        return Ok(false);
    }
    let shapes: Vec<(usize, PartialShape)> = targets
        .iter()
        .map(|(i, d)| {
            PartialShape::new_static(d.len() as i64, d)
                .map(|ps| (*i, ps))
                .with_context(|| format!("creating static shape {d:?}"))
        })
        .collect::<Result<_>>()?;
    let refs: Vec<(usize, &PartialShape)> = shapes.iter().map(|(i, s)| (*i, s)).collect();
    model.reshape_by_port_indexes(&refs).with_context(|| {
        format!(
            "model {}: reshaping dynamic inputs to {:?}",
            path.display(),
            targets
                .iter()
                .map(|(i, d)| (inputs[*i].name.as_str(), d))
                .collect::<Vec<_>>()
        )
    })?;
    tracing::info!(
        "model {}: reshaped dynamic inputs {:?}",
        path.display(),
        targets
            .iter()
            .map(|(i, d)| (inputs[*i].name.as_str(), d))
            .collect::<Vec<_>>()
    );
    Ok(true)
}

fn compile_on(
    core: &mut openvino::Core,
    model: &Model,
    device: &str,
    gpu_precision: &str,
    path: &std::path::Path,
) -> Result<openvino::CompiledModel> {
    let dt = DeviceType::from(device);
    if device.to_ascii_uppercase().starts_with("GPU") {
        core_set_property(
            core,
            &dt,
            &RwPropertyKey::HintInferencePrecision,
            gpu_precision,
        )
        .with_context(|| format!("setting INFERENCE_PRECISION_HINT={gpu_precision} on {device}"))?;
        tracing::info!("OpenVINO {device}: INFERENCE_PRECISION_HINT={gpu_precision}");
    }
    core.compile_model(model, dt)
        .with_context(|| format!("compiling model {} on {device}", path.display()))
}

/// With several GPUs (`GPU.0`, `GPU.1`, ...), OpenVINO's OpenCL plugin can also list non-Intel
/// cards (e.g. an NVIDIA dGPU). A bare `GPU` request is resolved to the first Intel GPU so it never
/// lands on an unsupported device. Explicit `GPU.N` requests are honoured as-is.
fn prefer_intel_gpu(core: &openvino::Core, primary: &str, available: &[String]) -> Option<String> {
    if !primary.eq_ignore_ascii_case("GPU") {
        return None;
    }
    let gpus: Vec<&String> = available.iter().filter(|d| d.starts_with("GPU.")).collect();
    if gpus.is_empty() {
        return None;
    }
    let default_name = full_device_name(core, "GPU");
    if default_name.contains("Intel") {
        return None;
    }
    gpus.into_iter()
        .find(|d| full_device_name(core, d).contains("Intel"))
        .cloned()
}

/// Read, reshape to static, compile on exactly `cand.device` (an OpenVINO device; a bare `GPU`
/// may resolve to the first Intel `GPU.N`) and introspect ports. No fallback here.
pub fn load_model(
    core: &mut openvino::Core,
    available: &[String],
    cand: &Candidate,
    req: &LoadRequest,
) -> Result<OvCompiled> {
    let mut primary = cand
        .device
        .openvino_device()
        .with_context(|| format!("{} is not an OpenVINO device", cand.device))?;
    let path = &req.path;
    if !path.is_file() {
        anyhow::bail!("model file {} does not exist", path.display());
    }
    let t0 = Instant::now();
    let mut model = read_model(core, path)?;
    let (inputs, _) =
        introspect(&model).with_context(|| format!("introspecting model {}", path.display()))?;
    let image_idx = find_image_input(&inputs).with_context(|| {
        format!(
            "model {}: no image input (named 'images'/'input' or 4-D) among {:?}",
            path.display(),
            inputs
                .iter()
                .map(|p| (&p.name, &p.shape))
                .collect::<Vec<_>>()
        )
    })?;
    let (inputs, outputs) = if make_static(&mut model, &inputs, image_idx, path)? {
        introspect(&model)
            .with_context(|| format!("introspecting reshaped model {}", path.display()))?
    } else {
        introspect(&model).with_context(|| format!("introspecting model {}", path.display()))?
    };
    let img = &inputs[image_idx];
    if img.shape.len() != 4 || img.shape.iter().any(|&d| d <= 0) {
        anyhow::bail!(
            "model {}: image input '{}' has non-static shape {:?} after reshape",
            path.display(),
            img.name,
            img.shape
        );
    }
    let input_size = (img.shape[3] as u32, img.shape[2] as u32);
    let image_input = img.name.clone();

    if let Some(intel) = prefer_intel_gpu(core, &primary, available) {
        tracing::info!(
            "model {}: device GPU resolved to {intel} (first Intel GPU; non-Intel OpenCL GPUs are skipped)",
            path.display()
        );
        primary = intel;
    }
    let precision = req
        .gpu_precision
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("f16")
        .to_ascii_lowercase();
    let compiled = compile_on(core, &model, &primary, &precision, path)?;
    let actual = primary;
    let compile_ms = t0.elapsed().as_millis() as u64;
    let device = DeviceInfo {
        requested: req.requested.clone(),
        full_name: full_device_name(core, &actual),
        actual,
        fell_back: cand.fell_back,
        runtime: Runtime::OpenVino,
        spec: cand.device.to_string(),
    };
    tracing::info!(
        "model {} compiled on {} ({}) in {compile_ms} ms; inputs {:?}, outputs {:?}",
        path.display(),
        device.actual,
        device.full_name,
        inputs
            .iter()
            .map(|p| (&p.name, &p.shape))
            .collect::<Vec<_>>(),
        outputs
            .iter()
            .map(|p| (&p.name, &p.shape))
            .collect::<Vec<_>>()
    );
    Ok(OvCompiled {
        compiled,
        info: ModelInfo {
            path: path.clone(),
            inputs,
            outputs,
            device,
            image_input,
            input_size,
            compile_ms,
        },
    })
}

fn image_index(model: &ModelInfo) -> Result<usize> {
    model
        .inputs
        .iter()
        .position(|p| p.name == model.image_input)
        .with_context(|| {
            format!(
                "model {}: image input '{}' not among inputs",
                model.path.display(),
                model.image_input
            )
        })
}

pub fn create_backend(compiled: OvCompiled) -> Result<OvBackend> {
    let OvCompiled {
        mut compiled,
        info: model,
    } = compiled;
    let idx = image_index(&model)?;
    let spec = &model.inputs[idx];
    if spec.elem != PortElem::F32 {
        anyhow::bail!(
            "model {}: image input '{}' has element type {:?}; only f32 is supported",
            model.path.display(),
            spec.name,
            spec.elem
        );
    }
    let shape = Shape::new(&spec.shape).with_context(|| {
        format!(
            "model {}: creating shape {:?} for input '{}'",
            model.path.display(),
            spec.shape,
            spec.name
        )
    })?;
    let input_tensor = Tensor::new(ElementType::F32, &shape).with_context(|| {
        format!(
            "model {}: allocating f32 input tensor {:?} for '{}'",
            model.path.display(),
            spec.shape,
            spec.name
        )
    })?;
    let request = compiled.create_infer_request().with_context(|| {
        format!(
            "model {}: creating infer request on {}",
            model.path.display(),
            model.device.actual
        )
    })?;
    Ok(OvBackend {
        info: model,
        request,
        input_tensor,
        _compiled: compiled,
    })
}

fn extra_tensor(model: &ModelInfo, port: &PortSpec, extra: &ExtraInput) -> Result<Tensor> {
    let ctx = || {
        format!(
            "model {}: extra input '{}' shape {:?}",
            model.path.display(),
            extra.name,
            extra.shape
        )
    };
    let dims: Vec<i64> = extra.shape.iter().map(|&d| d as i64).collect();
    let n: usize = extra.shape.iter().product();
    let len = match &extra.data {
        ExtraData::I64(v) => v.len(),
        ExtraData::I32(v) => v.len(),
        ExtraData::F32(v) => v.len(),
    };
    if len != n {
        anyhow::bail!("{}: {len} values for {n} elements", ctx());
    }
    let shape = Shape::new(&dims).with_context(ctx)?;
    // Convert to the port's element type when it differs from the supplied data.
    let target = match port.elem {
        PortElem::I64 => ElementType::I64,
        PortElem::I32 => ElementType::I32,
        PortElem::F32 => ElementType::F32,
        other => anyhow::bail!("{}: unsupported port element type {other:?}", ctx()),
    };
    let mut t = Tensor::new(target, &shape).with_context(ctx)?;
    match target {
        ElementType::I64 => {
            let dst = t.get_data_mut::<i64>().with_context(ctx)?;
            match &extra.data {
                ExtraData::I64(v) => dst.copy_from_slice(v),
                ExtraData::I32(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as i64),
                ExtraData::F32(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as i64),
            }
        }
        ElementType::I32 => {
            let dst = t.get_data_mut::<i32>().with_context(ctx)?;
            match &extra.data {
                ExtraData::I64(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as i32),
                ExtraData::I32(v) => dst.copy_from_slice(v),
                ExtraData::F32(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as i32),
            }
        }
        _ => {
            let dst = t.get_data_mut::<f32>().with_context(ctx)?;
            match &extra.data {
                ExtraData::I64(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as f32),
                ExtraData::I32(v) => dst.iter_mut().zip(v).for_each(|(d, s)| *d = *s as f32),
                ExtraData::F32(v) => dst.copy_from_slice(v),
            }
        }
    }
    Ok(t)
}

/// IEEE 754 half -> single precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) & 0x8000) << 16;
    let exp = (u32::from(h) >> 10) & 0x1f;
    let mant = u32::from(h) & 0x3ff;
    match exp {
        0 => {
            // Zero or subnormal: mant * 2^-24.
            let v = mant as f32 * (1.0 / 16_777_216.0);
            if sign != 0 { -v } else { v }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13)),
    }
}

fn read_output(model: &ModelInfo, tensor: &Tensor, spec: &PortSpec) -> Result<NamedOutput> {
    let ctx = || {
        format!(
            "model {} on {}: output '{}'",
            model.path.display(),
            model.device.actual,
            spec.name
        )
    };
    let shape: Vec<usize> = tensor
        .get_shape()
        .with_context(ctx)?
        .get_dimensions()
        .iter()
        .map(|&d| d.max(0) as usize)
        .collect();
    let elem = tensor.get_element_type().with_context(ctx)?;
    let n: usize = shape.iter().product();
    let data = match elem {
        ElementType::F32 => OutputBuf::F32(tensor.get_data::<f32>().with_context(ctx)?.to_vec()),
        ElementType::F16 => OutputBuf::F32(
            tensor
                .get_data::<u16>()
                .with_context(ctx)?
                .iter()
                .map(|&h| f16_to_f32(h))
                .collect(),
        ),
        ElementType::I64 => OutputBuf::I64(tensor.get_data::<i64>().with_context(ctx)?.to_vec()),
        ElementType::I32 => OutputBuf::I32(tensor.get_data::<i32>().with_context(ctx)?.to_vec()),
        other => anyhow::bail!("{}: unsupported element type {other}", ctx()),
    };
    let len = match &data {
        OutputBuf::F32(v) => v.len(),
        OutputBuf::I64(v) => v.len(),
        OutputBuf::I32(v) => v.len(),
    };
    if len != n {
        anyhow::bail!("{}: {len} values but shape {shape:?}", ctx());
    }
    Ok(NamedOutput {
        name: spec.name.clone(),
        shape,
        data,
    })
}

pub fn run_inference(
    model: &ModelInfo,
    request: &mut openvino::InferRequest,
    input_tensor: &mut openvino::Tensor,
    chw: &[f32],
    extra: &[ExtraInput],
) -> Result<Vec<NamedOutput>> {
    let where_ = || format!("model {} on {}", model.path.display(), model.device.actual);
    let idx = image_index(model)?;
    {
        let dst = input_tensor
            .get_data_mut::<f32>()
            .with_context(|| format!("{}: mapping input '{}'", where_(), model.image_input))?;
        if dst.len() != chw.len() {
            anyhow::bail!(
                "{}: input '{}' expects {} f32 values ({:?}), got {}",
                where_(),
                model.image_input,
                dst.len(),
                model.inputs[idx].shape,
                chw.len()
            );
        }
        dst.copy_from_slice(chw);
    }
    request
        .set_input_tensor_by_index(idx, input_tensor)
        .with_context(|| format!("{}: setting input '{}'", where_(), model.image_input))?;

    // Keep extra tensors alive until inference completes.
    let mut extras = Vec::with_capacity(extra.len());
    for e in extra {
        let i = extra_input_index(&model.inputs, idx, &e.name).with_context(|| {
            format!(
                "{}: extra input '{}' not among model inputs {:?}",
                where_(),
                e.name,
                model.inputs.iter().map(|p| &p.name).collect::<Vec<_>>()
            )
        })?;
        let t = extra_tensor(model, &model.inputs[i], e)?;
        request
            .set_input_tensor_by_index(i, &t)
            .with_context(|| format!("{}: setting input '{}'", where_(), e.name))?;
        extras.push(t);
    }

    request
        .infer()
        .with_context(|| format!("{}: inference failed", where_()))?;
    drop(extras);

    let mut outs = Vec::with_capacity(model.outputs.len());
    for (i, spec) in model.outputs.iter().enumerate() {
        let t = request
            .get_output_tensor_by_index(i)
            .with_context(|| format!("{}: getting output '{}' (#{i})", where_(), spec.name))?;
        outs.push(read_output(model, &t, spec)?);
    }
    Ok(outs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_provider_strings() {
        let mut info = DeviceInfo {
            requested: "GPU".into(),
            actual: "GPU".into(),
            full_name: "Intel(R) UHD Graphics 630".into(),
            fell_back: false,
            runtime: Runtime::OpenVino,
            spec: "openvino:gpu".into(),
        };
        assert!(info.is_gpu());
        assert_eq!(
            info.execution_provider(),
            "OpenVINO GPU (Intel(R) UHD Graphics 630)"
        );
        info.actual = "GPU.1".into();
        assert_eq!(
            info.execution_provider(),
            "OpenVINO GPU (Intel(R) UHD Graphics 630)"
        );
        info.actual = "CPU".into();
        info.full_name = "Apple M1".into();
        info.fell_back = true;
        info.spec = "openvino:cpu".into();
        assert!(!info.is_gpu());
        assert_eq!(
            info.execution_provider(),
            "OpenVINO CPU (Apple M1, fallback)"
        );
        info.full_name.clear();
        assert_eq!(info.execution_provider(), "OpenVINO CPU, fallback");
        info.fell_back = false;
        assert_eq!(info.execution_provider(), "OpenVINO CPU");

        let ort = DeviceInfo {
            requested: "ort:cuda".into(),
            actual: "cuda:0".into(),
            full_name: "NVIDIA GeForce RTX 3060".into(),
            fell_back: false,
            runtime: Runtime::Ort,
            spec: "ort:cuda:0".into(),
        };
        assert!(ort.is_gpu());
        assert_eq!(
            ort.execution_provider(),
            "ONNX Runtime CUDA (NVIDIA GeForce RTX 3060)"
        );
    }

    #[test]
    fn extra_input_lookup() {
        use crate::model::{PortElem, PortSpec};
        let port = |name: &str, shape: &[i64]| PortSpec {
            name: name.into(),
            shape: shape.to_vec(),
            elem: PortElem::F32,
        };
        let onnx = [
            port("images", &[1, 3, 640, 640]),
            port("orig_target_sizes", &[1, 2]),
        ];
        assert_eq!(extra_input_index(&onnx, 0, "orig_target_sizes"), Some(1));
        let ir = [
            port("images", &[1, 3, 640, 640]),
            port("/postprocessor/Expand_output_0", &[-1, 2]),
        ];
        assert_eq!(extra_input_index(&ir, 0, "orig_target_sizes"), Some(1));
        assert_eq!(extra_input_index(&ir[..1], 0, "orig_target_sizes"), None);
        let ambiguous = [
            port("a", &[1, 2]),
            port("images", &[1, 3, 640, 640]),
            port("b", &[1, 2]),
        ];
        assert_eq!(extra_input_index(&ambiguous, 1, "orig_target_sizes"), None);
        let not_2d = [
            port("images", &[1, 3, 640, 640]),
            port("mask", &[1, 1, 640, 640]),
        ];
        assert_eq!(extra_input_index(&not_2d, 0, "orig_target_sizes"), None);
    }

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn image_input_detection() {
        let p = |n: &str, s: &[i64]| PortSpec {
            name: n.into(),
            shape: s.to_vec(),
            elem: PortElem::F32,
        };
        assert_eq!(
            find_image_input(&[p("x", &[1, 2]), p("images", &[1, 3, 640, 640])]),
            Some(1)
        );
        assert_eq!(
            find_image_input(&[p("a", &[1, 2]), p("b", &[-1, 3, -1, -1])]),
            Some(1)
        );
        assert_eq!(find_image_input(&[p("a", &[1, 2])]), None);
    }
}
