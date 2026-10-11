//! Hardware detection without any runtime library (docs/PLAN.md, "Hardware detection").
//!
//! - **Windows**: DXGI `CreateDXGIFactory1` / `EnumAdapters1`: PCI vendor ID, name, dedicated VRAM;
//!   the software adapter (Microsoft Basic Render Driver) is skipped.
//! - **Linux**: `/sys/class/drm/cardN/device/{vendor,device}` (connector entries such as
//!   `card0-HDMI-A-1` are skipped), names from `/proc/driver/nvidia/gpus/*/information` when the
//!   NVIDIA driver is loaded, VRAM from amdgpu's `mem_info_vram_total`.
//! - **macOS arm64**: one Apple GPU entry (unified memory, so no VRAM figure).
//! - **NVIDIA** (Windows, Linux): the CUDA driver API (`nvcuda.dll` / `libcuda.so.1`, loaded at
//!   run time) adds each GPU's CUDA ordinal and compute capability, matched to the adapters by
//!   LUID (Windows) or PCI bus id (Linux), then by name. No driver, or any failing call, leaves
//!   them unknown.
//!
//! The result is computed once per process ([`hardware`]). The parsers are pure functions so they
//! are unit-tested with literal file contents.

use serde::{Serialize, Serializer};
use std::fmt;
use std::sync::OnceLock;

/// PCI vendor IDs.
pub const VENDOR_NVIDIA: u32 = 0x10DE;
pub const VENDOR_INTEL: u32 = 0x8086;
pub const VENDOR_AMD: u32 = 0x1002;
/// Microsoft (the DXGI software/"Basic Render Driver" adapter).
pub const VENDOR_MICROSOFT: u32 = 0x1414;

/// GPU vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    Nvidia,
    Intel,
    Amd,
    Apple,
    /// Any other PCI vendor ID.
    Other(u32),
}

impl GpuVendor {
    /// Vendor for a PCI vendor ID.
    pub fn from_pci_id(id: u32) -> Self {
        match id {
            VENDOR_NVIDIA => GpuVendor::Nvidia,
            VENDOR_INTEL => GpuVendor::Intel,
            VENDOR_AMD => GpuVendor::Amd,
            other => GpuVendor::Other(other),
        }
    }
}

impl fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuVendor::Nvidia => f.write_str("NVIDIA"),
            GpuVendor::Intel => f.write_str("Intel"),
            GpuVendor::Amd => f.write_str("AMD"),
            GpuVendor::Apple => f.write_str("Apple"),
            GpuVendor::Other(id) => write!(f, "vendor 0x{id:04x}"),
        }
    }
}

impl Serialize for GpuVendor {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// One hardware GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuAdapter {
    pub vendor: GpuVendor,
    pub name: String,
    /// Dedicated video memory in MiB; 0 when unknown or unified (iGPU, Apple).
    pub vram_mb: u64,
    /// Platform adapter index: the DXGI adapter index on Windows (what DirectML's `device_id`
    /// means), the DRM card number on Linux, 0 on macOS.
    pub index: u32,
    /// Discrete card (own VRAM) rather than integrated; a heuristic, see [`guess_discrete`].
    pub discrete: bool,
    /// What the CUDA driver reports for this (NVIDIA) GPU; None when unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda: Option<CudaInfo>,
}

/// CUDA compute capability, e.g. 6.1 (Pascal GP104) or 8.6 (Ampere GA106).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl Serialize for ComputeCapability {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// An NVIDIA GPU as the CUDA driver sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CudaInfo {
    /// CUDA device ordinal: what `ort:cuda:N` and ONNX Runtime's `device_id` mean (the driver's
    /// order, which honours `CUDA_VISIBLE_DEVICES` / `CUDA_DEVICE_ORDER`; not the DXGI index).
    pub ordinal: u32,
    pub compute_capability: ComputeCapability,
}

/// Detected hardware.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct HardwareInfo {
    /// Hardware GPUs in platform order (see [`GpuAdapter::index`]).
    pub gpus: Vec<GpuAdapter>,
    /// `std::env::consts::OS`: "windows", "linux", "macos", ...
    pub os: String,
    /// `std::env::consts::ARCH`: "x86_64", "aarch64", ...
    pub arch: String,
}

impl HardwareInfo {
    pub fn new(os: &str, arch: &str, gpus: Vec<GpuAdapter>) -> Self {
        Self {
            gpus,
            os: os.to_string(),
            arch: arch.to_string(),
        }
    }

    /// This platform with no GPUs (what the selection assumes when only OpenVINO's device list
    /// is known).
    pub fn this_platform_without_gpus() -> Self {
        Self::new(std::env::consts::OS, std::env::consts::ARCH, Vec::new())
    }

    pub fn is_windows(&self) -> bool {
        self.os == "windows"
    }

    pub fn is_macos(&self) -> bool {
        self.os == "macos"
    }

    /// macOS on Apple silicon (where CoreML is worth picking automatically).
    pub fn is_apple_silicon(&self) -> bool {
        self.is_macos() && self.arch == "aarch64"
    }

    pub fn gpus_of(&self, vendor: GpuVendor) -> impl Iterator<Item = &GpuAdapter> {
        self.gpus.iter().filter(move |g| g.vendor == vendor)
    }

    pub fn has_vendor(&self, vendor: GpuVendor) -> bool {
        self.gpus_of(vendor).next().is_some()
    }
}

impl fmt::Display for GpuAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{} {} ({}", self.index, self.name, self.vendor)?;
        f.write_str(if self.discrete {
            ", discrete"
        } else {
            ", integrated"
        })?;
        if self.vram_mb > 0 {
            write!(f, ", {} MB VRAM", self.vram_mb)?;
        }
        if let Some(c) = &self.cuda {
            write!(
                f,
                ", CUDA device {} compute capability {}",
                c.ordinal, c.compute_capability
            )?;
        }
        f.write_str(")")
    }
}

/// Detected hardware, computed on first use and cached for the life of the process.
pub fn hardware() -> &'static HardwareInfo {
    static HW: OnceLock<HardwareInfo> = OnceLock::new();
    HW.get_or_init(detect)
}

/// Detect GPUs now (prefer the cached [`hardware`]). Never fails: errors are logged at debug
/// level and yield fewer GPUs.
pub fn detect() -> HardwareInfo {
    let t = std::time::Instant::now();
    let (mut gpus, keys) = detect_gpus();
    if gpus.iter().any(|g| g.vendor == GpuVendor::Nvidia) {
        let cuda = cuda_driver::devices();
        attach_cuda(&mut gpus, &keys, &cuda);
    }
    let hw = HardwareInfo::new(std::env::consts::OS, std::env::consts::ARCH, gpus);
    tracing::debug!(
        gpus = ?hw.gpus,
        ms = t.elapsed().as_millis() as u64,
        "hardware detection"
    );
    hw
}

/// GPUs plus, per GPU, the key that identifies it to the CUDA driver.
type Detected = (Vec<GpuAdapter>, Vec<AdapterKey>);

#[cfg(windows)]
fn detect_gpus() -> Detected {
    match dxgi::adapters() {
        Ok(v) => v
            .into_iter()
            .map(|(g, luid)| (g, AdapterKey::Luid(luid)))
            .unzip(),
        Err(e) => {
            tracing::debug!("DXGI adapter enumeration failed: {e:#}");
            (Vec::new(), Vec::new())
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_gpus() -> Detected {
    linux::adapters_with_slots(
        std::path::Path::new("/sys/class/drm"),
        std::path::Path::new("/proc/driver/nvidia/gpus"),
    )
    .into_iter()
    .map(|(g, slot)| {
        let key = slot
            .as_deref()
            .and_then(PciAddress::parse)
            .map_or(AdapterKey::None, AdapterKey::Pci);
        (g, key)
    })
    .unzip()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn detect_gpus() -> Detected {
    (
        vec![apple_gpu(&crate::system_info::cpu_name())],
        vec![AdapterKey::None],
    )
}

#[cfg(not(any(
    windows,
    target_os = "linux",
    all(target_os = "macos", target_arch = "aarch64")
)))]
fn detect_gpus() -> Detected {
    (Vec::new(), Vec::new())
}

/// PCI location (domain, bus, device, function).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PciAddress {
    pub domain: u32,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

impl PciAddress {
    /// `0000:01:00.0` (sysfs) or `00000000:01:00.0` (CUDA/NVML, 8-digit domain); case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_end_matches('\0');
        let (rest, function) = s.rsplit_once('.')?;
        let mut it = rest.split(':');
        let (domain, bus, device) = (it.next()?, it.next()?, it.next()?);
        if it.next().is_some() {
            return None;
        }
        let hex = |p: &str| (!p.is_empty()).then(|| u32::from_str_radix(p, 16).ok())?;
        Some(Self {
            domain: hex(domain)?,
            bus: hex(bus)?,
            device: hex(device)?,
            function: hex(function)?,
        })
    }
}

/// How the platform identifies an adapter to the CUDA driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKey {
    /// Windows adapter LUID (`DXGI_ADAPTER_DESC1::AdapterLuid`, LowPart in the low 32 bits).
    Luid(u64),
    /// Linux PCI slot.
    Pci(PciAddress),
    None,
}

/// One device the CUDA driver lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDevice {
    pub ordinal: u32,
    /// `cuDeviceGetName`, e.g. "NVIDIA GeForce GTX 1070 Ti".
    pub name: String,
    pub compute_capability: ComputeCapability,
    /// `cuDeviceGetLuid` (Windows only).
    pub luid: Option<u64>,
    /// `cuDeviceGetPCIBusId`.
    pub pci: Option<PciAddress>,
}

/// Set [`GpuAdapter::cuda`] on the NVIDIA adapters in `gpus` (`keys[i]` identifies `gpus[i]`).
/// Matching, each CUDA device used at most once: by LUID / PCI address; then, in platform
/// order, by name; then a single adapter left with a single device left. Adapters that do not
/// match stay unknown. Pure.
pub fn attach_cuda(gpus: &mut [GpuAdapter], keys: &[AdapterKey], cuda: &[CudaDevice]) {
    let nvidia: Vec<usize> = (0..gpus.len())
        .filter(|&i| gpus[i].vendor == GpuVendor::Nvidia)
        .collect();
    let mut used = vec![false; cuda.len()];
    let mut pick = vec![None::<usize>; gpus.len()];
    // First unused device matching `f`, marked used.
    let mut claim = |f: &dyn Fn(&CudaDevice) -> bool| {
        let j = (0..cuda.len()).find(|&j| !used[j] && f(&cuda[j]))?;
        used[j] = true;
        Some(j)
    };
    for &i in &nvidia {
        pick[i] = match keys.get(i).copied().unwrap_or(AdapterKey::None) {
            AdapterKey::Luid(l) => claim(&|c| c.luid == Some(l)),
            AdapterKey::Pci(p) => claim(&|c| c.pci == Some(p)),
            AdapterKey::None => None,
        };
    }
    for &i in &nvidia {
        if pick[i].is_none() {
            let name = gpus[i].name.trim();
            pick[i] = claim(&|c| c.name.trim().eq_ignore_ascii_case(name));
        }
    }
    let open: Vec<usize> = nvidia
        .iter()
        .copied()
        .filter(|&i| pick[i].is_none())
        .collect();
    let free: Vec<usize> = (0..cuda.len()).filter(|&j| !used[j]).collect();
    if let ([i], [j]) = (open.as_slice(), free.as_slice()) {
        pick[*i] = Some(*j);
    }
    for (g, j) in gpus.iter_mut().zip(pick) {
        g.cuda = j.map(|j| CudaInfo {
            ordinal: cuda[j].ordinal,
            compute_capability: cuda[j].compute_capability,
        });
    }
}

/// The single Apple silicon GPU, named after the chip ("Apple M2 Pro" -> "Apple M2 Pro GPU").
pub fn apple_gpu(chip: &str) -> GpuAdapter {
    let chip = chip.trim();
    let name = if chip.starts_with("Apple") {
        format!("{chip} GPU")
    } else {
        "Apple GPU".to_string()
    };
    GpuAdapter {
        vendor: GpuVendor::Apple,
        name,
        vram_mb: 0,
        index: 0,
        discrete: false,
        cuda: None,
    }
}

/// Intel discrete GPU PCI device IDs: DG1 (0x4905-0x4909), Alchemist/DG2 Arc A-series and Flex
/// (0x5690-0x56FF), Ponte Vecchio (0x0BD0-0x0BDF), Battlemage Arc B-series (0xE200-0xE2FF).
pub fn intel_device_is_discrete(device_id: u32) -> bool {
    matches!(
        device_id,
        0x4905..=0x4909 | 0x5690..=0x56FF | 0x0BD0..=0x0BDF | 0xE200..=0xE2FF
    )
}

/// Discrete vs integrated without a runtime. NVIDIA is always discrete; Intel by device ID (or a
/// large dedicated VRAM); AMD and others when they report more than 1 GiB of dedicated VRAM (APUs
/// report their small BIOS carve-out, typically 512 MiB); Apple never.
pub fn guess_discrete(vendor: GpuVendor, device_id: u32, vram_mb: u64) -> bool {
    match vendor {
        GpuVendor::Nvidia => true,
        GpuVendor::Apple => false,
        GpuVendor::Intel => intel_device_is_discrete(device_id) || vram_mb > 1024,
        GpuVendor::Amd | GpuVendor::Other(_) => vram_mb > 1024,
    }
}

/// Fallback adapter name when the platform gives none: "Intel GPU [8086:3e92]".
pub fn generic_name(vendor_id: u32, device_id: u32) -> String {
    let vendor = match GpuVendor::from_pci_id(vendor_id) {
        GpuVendor::Other(_) => "Unknown".to_string(),
        v => v.to_string(),
    };
    format!("{vendor} GPU [{vendor_id:04x}:{device_id:04x}]")
}

/// Build an adapter from raw PCI facts. `name` falls back to [`generic_name`].
pub fn adapter_from_ids(
    index: u32,
    vendor_id: u32,
    device_id: u32,
    name: Option<&str>,
    vram_mb: u64,
) -> GpuAdapter {
    let vendor = GpuVendor::from_pci_id(vendor_id);
    let name = name
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| generic_name(vendor_id, device_id));
    GpuAdapter {
        vendor,
        name,
        vram_mb,
        index,
        discrete: guess_discrete(vendor, device_id, vram_mb),
        cuda: None,
    }
}

/// UTF-16 DXGI description (NUL padded) to a String.
pub fn utf16_name(raw: &[u16]) -> String {
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..end]).trim().to_string()
}

/// `0x8086\n` (sysfs `vendor`/`device`) or `8086` to a number.
pub fn parse_hex_id(s: &str) -> Option<u32> {
    let s = s.trim();
    let digits = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    if digits.is_empty() {
        return None;
    }
    u32::from_str_radix(digits, 16).ok()
}

/// Card number of a `/sys/class/drm` entry: `card1` -> 1; connectors (`card0-HDMI-A-1`),
/// render nodes (`renderD128`) and anything else -> None.
pub fn drm_card_index(entry: &str) -> Option<u32> {
    let digits = entry.strip_prefix("card")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// `PCI_SLOT_NAME` from a sysfs `uevent` file, lowercased ("0000:01:00.0").
pub fn uevent_pci_slot(uevent: &str) -> Option<String> {
    uevent
        .lines()
        .find_map(|l| l.trim().strip_prefix("PCI_SLOT_NAME="))
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
}

/// What `/proc/driver/nvidia/gpus/<bus>/information` tells us.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaProcInfo {
    /// "NVIDIA GeForce RTX 3060".
    pub model: String,
    /// Lowercased PCI bus location ("0000:01:00.0"), when present.
    pub bus: Option<String>,
}

/// Parse an NVIDIA `information` file (`Key:  value` lines). None without a `Model:` line.
pub fn parse_nvidia_information(text: &str) -> Option<NvidiaProcInfo> {
    let value = |key: &str| {
        text.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            (k.trim() == key).then(|| v.trim().to_string())
        })
    };
    let model = value("Model").filter(|m| !m.is_empty())?;
    let bus = value("Bus Location")
        .map(|b| b.to_ascii_lowercase())
        .filter(|b| !b.is_empty());
    Some(NvidiaProcInfo { model, bus })
}

/// Linux sysfs/procfs scanning. The file system roots are parameters so this is testable on a
/// temporary directory tree.
#[cfg(any(target_os = "linux", test))]
pub(crate) mod linux {
    use super::*;
    use std::path::Path;

    fn read(p: &Path) -> Option<String> {
        std::fs::read_to_string(p).ok()
    }

    /// NVIDIA `information` files under `nvidia_root`, sorted by directory (bus) name.
    fn nvidia_infos(nvidia_root: &Path) -> Vec<(String, NvidiaProcInfo)> {
        let Ok(dir) = std::fs::read_dir(nvidia_root) else {
            return Vec::new();
        };
        let mut v: Vec<(String, NvidiaProcInfo)> = dir
            .flatten()
            .filter_map(|e| {
                let dir_name = e.file_name().to_string_lossy().to_ascii_lowercase();
                let info = parse_nvidia_information(&read(&e.path().join("information"))?)?;
                Some((dir_name, info))
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// GPUs from `drm_root` (`/sys/class/drm`) plus names from `nvidia_root`
    /// (`/proc/driver/nvidia/gpus`). NVIDIA GPUs the driver lists without a DRM card (nvidia-drm
    /// not loaded) are appended after the cards.
    pub fn adapters(drm_root: &Path, nvidia_root: &Path) -> Vec<GpuAdapter> {
        adapters_with_slots(drm_root, nvidia_root)
            .into_iter()
            .map(|(g, _)| g)
            .collect()
    }

    /// [`adapters`] with each GPU's PCI slot ("0000:01:00.0"), when known.
    pub fn adapters_with_slots(
        drm_root: &Path,
        nvidia_root: &Path,
    ) -> Vec<(GpuAdapter, Option<String>)> {
        let mut nvidia = nvidia_infos(nvidia_root);
        let mut cards: Vec<(u32, std::path::PathBuf)> = std::fs::read_dir(drm_root)
            .map(|d| {
                d.flatten()
                    .filter_map(|e| {
                        let idx = drm_card_index(&e.file_name().to_string_lossy())?;
                        Some((idx, e.path()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        cards.sort_by_key(|(i, _)| *i);

        let mut out = Vec::new();
        for (idx, path) in cards {
            let dev = path.join("device");
            // Virtual/platform DRM devices (simpledrm, vkms) have no PCI ids.
            let (Some(vendor_id), Some(device_id)) = (
                read(&dev.join("vendor")).as_deref().and_then(parse_hex_id),
                read(&dev.join("device")).as_deref().and_then(parse_hex_id),
            ) else {
                continue;
            };
            let slot = read(&dev.join("uevent"))
                .as_deref()
                .and_then(uevent_pci_slot);
            let mut name = None;
            if vendor_id == VENDOR_NVIDIA {
                let pos = nvidia
                    .iter()
                    .position(|(dir, info)| {
                        slot.as_deref()
                            .is_some_and(|s| dir == s || info.bus.as_deref() == Some(s))
                    })
                    .or(if nvidia.len() == 1 { Some(0) } else { None });
                if let Some(pos) = pos {
                    name = Some(nvidia.remove(pos).1.model);
                }
            }
            // amdgpu reports dedicated VRAM in bytes.
            let vram_mb = read(&dev.join("mem_info_vram_total"))
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| b / (1024 * 1024))
                .unwrap_or(0);
            out.push((
                adapter_from_ids(idx, vendor_id, device_id, name.as_deref(), vram_mb),
                slot,
            ));
        }
        let first = out.iter().map(|(g, _)| g.index + 1).max().unwrap_or(0);
        for (index, (_, info)) in (first..).zip(nvidia) {
            out.push((
                adapter_from_ids(index, VENDOR_NVIDIA, 0, Some(&info.model), 0),
                info.bus,
            ));
        }
        out
    }
}

/// DXGI adapter enumeration (Windows only; one of the two places in this module with `unsafe`).
#[cfg(windows)]
mod dxgi {
    use super::*;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, IDXGIFactory1,
    };

    /// Hardware adapters with their LUIDs (`LowPart | HighPart << 32`).
    pub fn adapters() -> anyhow::Result<Vec<(GpuAdapter, u64)>> {
        // SAFETY: plain COM factory creation; the returned interface is reference counted.
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }?;
        let mut out = Vec::new();
        for i in 0u32.. {
            // SAFETY: `factory` is a valid IDXGIFactory1; DXGI_ERROR_NOT_FOUND ends the list.
            let adapter = match unsafe { factory.EnumAdapters1(i) } {
                Ok(a) => a,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(e.into()),
            };
            // SAFETY: `adapter` is a valid IDXGIAdapter1.
            let desc = match unsafe { adapter.GetDesc1() } {
                Ok(d) => d,
                Err(e) => {
                    tracing::debug!("DXGI adapter {i}: GetDesc1 failed: {e}");
                    continue;
                }
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                || desc.VendorId == VENDOR_MICROSOFT
            {
                continue;
            }
            let vram_mb = desc.DedicatedVideoMemory as u64 / (1024 * 1024);
            let name = utf16_name(&desc.Description);
            let luid =
                (desc.AdapterLuid.HighPart as u32 as u64) << 32 | desc.AdapterLuid.LowPart as u64;
            out.push((
                adapter_from_ids(i, desc.VendorId, desc.DeviceId, Some(&name), vram_mb),
                luid,
            ));
        }
        Ok(out)
    }
}

/// CUDA driver API queries (`nvcuda.dll` / `libcuda.so.1`, loaded at run time; the other place
/// in this module with `unsafe`). Read-only: no context is created.
#[cfg(any(windows, target_os = "linux"))]
mod cuda_driver {
    use super::*;
    use std::ffi::{CStr, c_char, c_int, c_uint};

    type CuResult = c_int;
    type CuDevice = c_int;
    const CUDA_SUCCESS: CuResult = 0;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

    #[cfg(windows)]
    const LIBRARY: &str = "nvcuda.dll";
    #[cfg(not(windows))]
    const LIBRARY: &str = "libcuda.so.1";

    // CUDAAPI is __stdcall on Windows (the C convention on x64), cdecl elsewhere: "system".
    type CuInit = unsafe extern "system" fn(c_uint) -> CuResult;
    type CuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "system" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetAttribute = unsafe extern "system" fn(*mut c_int, c_int, CuDevice) -> CuResult;
    type CuDeviceGetString = unsafe extern "system" fn(*mut c_char, c_int, CuDevice) -> CuResult;
    #[cfg(windows)]
    type CuDeviceGetLuid =
        unsafe extern "system" fn(*mut c_char, *mut c_uint, CuDevice) -> CuResult;

    /// Every device the driver lists; empty (logged at debug level) when there is no driver or
    /// any call fails.
    pub fn devices() -> Vec<CudaDevice> {
        match query() {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!("CUDA driver query failed, compute capability unknown: {e}");
                Vec::new()
            }
        }
    }

    fn check(what: &str, r: CuResult) -> Result<(), String> {
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what} returned CUresult {r}"))
        }
    }

    fn c_string(buf: &[c_char]) -> String {
        let bytes: Vec<u8> = buf.iter().map(|&c| c as u8).collect();
        CStr::from_bytes_until_nul(&bytes)
            .map(|s| s.to_string_lossy().trim().to_string())
            .unwrap_or_default()
    }

    fn query() -> Result<Vec<CudaDevice>, String> {
        // SAFETY: loads the NVIDIA driver library, the same one ONNX Runtime's CUDA provider
        // loads; its initializers have no preconditions.
        let lib =
            unsafe { libloading::Library::new(LIBRARY) }.map_err(|e| format!("{LIBRARY}: {e}"))?;
        // Never unloaded: unloading the driver after cuInit is not supported, and the CUDA
        // provider uses it later anyway.
        let lib: &'static libloading::Library = Box::leak(Box::new(lib));
        // SAFETY: the symbol types match the CUDA driver API (cuda.h) declarations.
        let sym = |name: &[u8]| -> Result<*const (), String> {
            unsafe { lib.get::<*const ()>(name) }
                .map(|s| *s)
                .map_err(|e| format!("{LIBRARY}: {e}"))
        };
        // SAFETY: each pointer comes from the symbol of that name and is transmuted to its
        // cuda.h signature.
        let (init, count, get, attr, name, bus) = unsafe {
            (
                std::mem::transmute::<*const (), CuInit>(sym(b"cuInit\0")?),
                std::mem::transmute::<*const (), CuDeviceGetCount>(sym(b"cuDeviceGetCount\0")?),
                std::mem::transmute::<*const (), CuDeviceGet>(sym(b"cuDeviceGet\0")?),
                std::mem::transmute::<*const (), CuDeviceGetAttribute>(sym(
                    b"cuDeviceGetAttribute\0",
                )?),
                std::mem::transmute::<*const (), CuDeviceGetString>(sym(b"cuDeviceGetName\0")?),
                std::mem::transmute::<*const (), CuDeviceGetString>(sym(b"cuDeviceGetPCIBusId\0")?),
            )
        };
        #[cfg(windows)]
        // SAFETY: as above.
        let luid_fn = sym(b"cuDeviceGetLuid\0")
            .ok()
            .map(|p| unsafe { std::mem::transmute::<*const (), CuDeviceGetLuid>(p) });

        // SAFETY (all calls below): plain driver queries writing into local out-parameters of
        // the documented sizes.
        check("cuInit", unsafe { init(0) })?;
        let mut n: c_int = 0;
        check("cuDeviceGetCount", unsafe { count(&mut n) })?;
        let mut out = Vec::new();
        for ordinal in 0..n.max(0) {
            let mut dev: CuDevice = 0;
            check("cuDeviceGet", unsafe { get(&mut dev, ordinal) })?;
            let (mut major, mut minor): (c_int, c_int) = (0, 0);
            check("cuDeviceGetAttribute(major)", unsafe {
                attr(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                )
            })?;
            check("cuDeviceGetAttribute(minor)", unsafe {
                attr(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                )
            })?;
            let mut name_buf = [0 as c_char; 256];
            check("cuDeviceGetName", unsafe {
                name(name_buf.as_mut_ptr(), name_buf.len() as c_int, dev)
            })?;
            let mut bus_buf = [0 as c_char; 64];
            let pci = (unsafe { bus(bus_buf.as_mut_ptr(), bus_buf.len() as c_int, dev) }
                == CUDA_SUCCESS)
                .then(|| PciAddress::parse(&c_string(&bus_buf)))
                .flatten();
            #[cfg(windows)]
            let luid = luid_fn.and_then(|f| {
                let mut raw = [0 as c_char; 8];
                let mut mask: c_uint = 0;
                (unsafe { f(raw.as_mut_ptr(), &mut mask, dev) } == CUDA_SUCCESS)
                    .then(|| u64::from_le_bytes(raw.map(|c| c as u8)))
            });
            #[cfg(not(windows))]
            let luid = None;
            out.push(CudaDevice {
                ordinal: ordinal as u32,
                name: c_string(&name_buf),
                compute_capability: ComputeCapability::new(
                    major.max(0) as u32,
                    minor.max(0) as u32,
                ),
                luid,
                pci,
            });
        }
        tracing::debug!(devices = ?out, "CUDA driver devices");
        Ok(out)
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod cuda_driver {
    /// No CUDA on this platform.
    pub fn devices() -> Vec<super::CudaDevice> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn vendor_mapping() {
        assert_eq!(GpuVendor::from_pci_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_pci_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_pci_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_pci_id(0x5143), GpuVendor::Other(0x5143));
        assert_eq!(GpuVendor::Other(0x5143).to_string(), "vendor 0x5143");
        assert_eq!(
            serde_json::to_string(&GpuVendor::Nvidia).unwrap(),
            "\"NVIDIA\""
        );
    }

    #[test]
    fn hex_ids_and_card_names() {
        assert_eq!(parse_hex_id("0x8086\n"), Some(0x8086));
        assert_eq!(parse_hex_id("10de"), Some(0x10DE));
        assert_eq!(parse_hex_id("0X1002"), Some(0x1002));
        assert_eq!(parse_hex_id(""), None);
        assert_eq!(parse_hex_id("0x"), None);
        assert_eq!(parse_hex_id("0xzz"), None);

        assert_eq!(drm_card_index("card0"), Some(0));
        assert_eq!(drm_card_index("card12"), Some(12));
        assert_eq!(drm_card_index("card0-HDMI-A-1"), None);
        assert_eq!(drm_card_index("card1-eDP-1"), None);
        assert_eq!(drm_card_index("renderD128"), None);
        assert_eq!(drm_card_index("card"), None);
        assert_eq!(drm_card_index("version"), None);
    }

    #[test]
    fn uevent_and_nvidia_information() {
        let uevent = "DRIVER=nvidia\nPCI_CLASS=30000\nPCI_ID=10DE:2504\n\
                      PCI_SLOT_NAME=0000:01:00.0\nMODALIAS=pci:v000010DEd00002504\n";
        assert_eq!(uevent_pci_slot(uevent).as_deref(), Some("0000:01:00.0"));
        assert_eq!(uevent_pci_slot("DRIVER=i915\n"), None);

        let info = "Model: \t\t NVIDIA GeForce RTX 3060\n\
                    IRQ:   \t\t 140\n\
                    GPU UUID: \t GPU-6a1b2c3d-0000-1111-2222-333344445555\n\
                    Video BIOS: \t 94.06.2f.00.9a\n\
                    Bus Type: \t PCIe\n\
                    DMA Size: \t 47 bits\n\
                    Bus Location: \t 0000:01:00.0\n\
                    Device Minor: \t 0\n";
        assert_eq!(
            parse_nvidia_information(info),
            Some(NvidiaProcInfo {
                model: "NVIDIA GeForce RTX 3060".into(),
                bus: Some("0000:01:00.0".into()),
            })
        );
        assert_eq!(parse_nvidia_information("IRQ: 1\n"), None);
        assert_eq!(parse_nvidia_information("Model:   \n"), None);
    }

    #[test]
    fn discrete_heuristics() {
        // UHD 630 (0x3E92) with its 128 MB DXGI carve-out: integrated.
        assert!(!guess_discrete(GpuVendor::Intel, 0x3E92, 128));
        assert!(guess_discrete(GpuVendor::Intel, 0x56A0, 0)); // Arc A770
        assert!(guess_discrete(GpuVendor::Intel, 0xE20B, 0)); // Arc B580
        assert!(guess_discrete(GpuVendor::Intel, 0x4905, 0)); // DG1
        assert!(!guess_discrete(GpuVendor::Intel, 0x7D55, 0)); // Meteor Lake Arc iGPU
        assert!(guess_discrete(GpuVendor::Intel, 0x1234, 8192));
        assert!(guess_discrete(GpuVendor::Nvidia, 0, 0));
        assert!(guess_discrete(GpuVendor::Amd, 0x73FF, 8176));
        assert!(!guess_discrete(GpuVendor::Amd, 0x1638, 512)); // APU carve-out
        assert!(!guess_discrete(GpuVendor::Apple, 0, 0));
    }

    #[test]
    fn adapter_assembly_and_names() {
        let g = adapter_from_ids(1, 0x10DE, 0x2504, Some(" NVIDIA GeForce RTX 3060 "), 12288);
        assert_eq!(
            g,
            GpuAdapter {
                vendor: GpuVendor::Nvidia,
                name: "NVIDIA GeForce RTX 3060".into(),
                vram_mb: 12288,
                index: 1,
                discrete: true,
                cuda: None,
            }
        );
        let g = adapter_from_ids(0, 0x8086, 0x3E92, None, 0);
        assert_eq!(g.name, "Intel GPU [8086:3e92]");
        assert!(!g.discrete);
        assert_eq!(generic_name(0x1AF4, 0x1050), "Unknown GPU [1af4:1050]");

        let mut raw = [0u16; 128];
        for (i, c) in "Intel(R) UHD Graphics 630".encode_utf16().enumerate() {
            raw[i] = c;
        }
        assert_eq!(utf16_name(&raw), "Intel(R) UHD Graphics 630");
        assert_eq!(utf16_name(&[]), "");

        assert_eq!(apple_gpu("Apple M2 Pro").name, "Apple M2 Pro GPU");
        assert_eq!(apple_gpu("arm64").name, "Apple GPU");
        assert_eq!(apple_gpu("x").vendor, GpuVendor::Apple);
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn linux_sysfs_tree() {
        let root = std::env::temp_dir().join(format!(
            "bop-detect-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let drm = root.join("drm");
        let nv = root.join("nvidia");
        // card0: Intel iGPU, card1: NVIDIA, card2: virtual (no ids), plus connector/render nodes.
        write(&drm.join("card0/device/vendor"), "0x8086\n");
        write(&drm.join("card0/device/device"), "0x3e92\n");
        write(&drm.join("card0-HDMI-A-1/status"), "connected\n");
        write(&drm.join("renderD128/dev"), "226:128\n");
        write(&drm.join("card1/device/vendor"), "0x10de\n");
        write(&drm.join("card1/device/device"), "0x2504\n");
        write(
            &drm.join("card1/device/uevent"),
            "DRIVER=nvidia\nPCI_SLOT_NAME=0000:01:00.0\n",
        );
        write(
            &drm.join("card2/device/uevent"),
            "DRIVER=simple-framebuffer\n",
        );
        write(&drm.join("card3/device/vendor"), "0x1002\n");
        write(&drm.join("card3/device/device"), "0x73ff\n");
        write(
            &drm.join("card3/device/mem_info_vram_total"),
            "8573157376\n",
        );
        // Real directories are named after the bus ("0000:01:00.0"), which is not a valid file
        // name on Windows; these are matched through their `Bus Location` line instead.
        write(
            &nv.join("gpu-01/information"),
            "Model: \t\t NVIDIA GeForce RTX 3060\nBus Location: \t 0000:01:00.0\n",
        );
        // A second NVIDIA GPU without a DRM card.
        write(
            &nv.join("gpu-02/information"),
            "Model: \t\t NVIDIA RTX A2000\nBus Location: \t 0000:02:00.0\n",
        );

        let gpus = linux::adapters(&drm, &nv);
        let summary: Vec<(u32, GpuVendor, &str, u64, bool)> = gpus
            .iter()
            .map(|g| (g.index, g.vendor, g.name.as_str(), g.vram_mb, g.discrete))
            .collect();
        assert_eq!(
            summary,
            vec![
                (0, GpuVendor::Intel, "Intel GPU [8086:3e92]", 0, false),
                (1, GpuVendor::Nvidia, "NVIDIA GeForce RTX 3060", 0, true),
                (3, GpuVendor::Amd, "AMD GPU [1002:73ff]", 8176, true),
                (4, GpuVendor::Nvidia, "NVIDIA RTX A2000", 0, true),
            ]
        );
        // Missing roots: nothing, no panic.
        assert!(linux::adapters(&root.join("none"), &root.join("none")).is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pci_addresses() {
        let p = PciAddress {
            domain: 0,
            bus: 1,
            device: 0,
            function: 0,
        };
        assert_eq!(PciAddress::parse("0000:01:00.0"), Some(p));
        assert_eq!(PciAddress::parse("00000000:01:00.0\0\0"), Some(p));
        assert_eq!(
            PciAddress::parse("0001:AF:1f.7").map(|a| (a.domain, a.bus, a.device, a.function)),
            Some((1, 0xAF, 0x1F, 7))
        );
        assert_eq!(PciAddress::parse("01:00.0"), None);
        assert_eq!(PciAddress::parse("0000:01:00"), None);
        assert_eq!(PciAddress::parse("0000::00.0"), None);
        assert_eq!(PciAddress::parse(""), None);
    }

    fn cuda_dev(ordinal: u32, name: &str, cc: (u32, u32), luid: Option<u64>) -> CudaDevice {
        CudaDevice {
            ordinal,
            name: name.into(),
            compute_capability: ComputeCapability::new(cc.0, cc.1),
            luid,
            pci: None,
        }
    }

    fn ordinals(gpus: &[GpuAdapter]) -> Vec<Option<(u32, String)>> {
        gpus.iter()
            .map(|g| {
                g.cuda
                    .map(|c| (c.ordinal, c.compute_capability.to_string()))
            })
            .collect()
    }

    #[test]
    fn cuda_devices_attach_by_key_name_or_last_one() {
        let intel = || {
            adapter_from_ids(
                1,
                VENDOR_INTEL,
                0x3E92,
                Some("Intel(R) UHD Graphics 630"),
                128,
            )
        };
        let nv = |i: u32, name: &str| adapter_from_ids(i, VENDOR_NVIDIA, 0x1B82, Some(name), 8060);

        // LUID wins over name and order: identical names, the driver lists them the other way.
        let mut gpus = vec![nv(0, "NVIDIA X"), intel(), nv(2, "NVIDIA X")];
        let keys = [
            AdapterKey::Luid(10),
            AdapterKey::Luid(11),
            AdapterKey::Luid(12),
        ];
        let cuda = [
            cuda_dev(0, "NVIDIA X", (8, 6), Some(12)),
            cuda_dev(1, "NVIDIA X", (6, 1), Some(10)),
        ];
        attach_cuda(&mut gpus, &keys, &cuda);
        assert_eq!(
            ordinals(&gpus),
            [Some((1, "6.1".into())), None, Some((0, "8.6".into()))]
        );

        // PCI address (Linux).
        let addr = |bus| PciAddress {
            domain: 0,
            bus,
            device: 0,
            function: 0,
        };
        let mut gpus = vec![nv(0, "A"), nv(1, "B")];
        let keys = [AdapterKey::Pci(addr(1)), AdapterKey::Pci(addr(2))];
        let mut c0 = cuda_dev(0, "B", (8, 9), None);
        c0.pci = Some(addr(2));
        let mut c1 = cuda_dev(1, "A", (7, 5), None);
        c1.pci = Some(addr(1));
        attach_cuda(&mut gpus, &keys, &[c0, c1]);
        assert_eq!(
            ordinals(&gpus),
            [Some((1, "7.5".into())), Some((0, "8.9".into()))]
        );

        // No keys: by name; the one left over pairs with the one device left.
        let mut gpus = vec![
            nv(0, "NVIDIA GeForce GTX 1070 Ti"),
            nv(1, "Odd name"),
            intel(),
        ];
        let cuda = [
            cuda_dev(0, "Other", (8, 6), None),
            cuda_dev(1, "nvidia geforce gtx 1070 ti", (6, 1), None),
        ];
        attach_cuda(&mut gpus, &[], &cuda);
        assert_eq!(
            ordinals(&gpus),
            [Some((1, "6.1".into())), Some((0, "8.6".into())), None]
        );

        // Ambiguous leftovers stay unknown; no driver devices -> unknown.
        let mut gpus = vec![nv(0, "P"), nv(1, "Q")];
        let cuda = [
            cuda_dev(0, "R", (8, 6), None),
            cuda_dev(1, "S", (8, 6), None),
        ];
        attach_cuda(&mut gpus, &[], &cuda);
        assert_eq!(ordinals(&gpus), [None, None]);
        let mut gpus = vec![nv(0, "P")];
        attach_cuda(&mut gpus, &[AdapterKey::Luid(1)], &[]);
        assert_eq!(ordinals(&gpus), [None]);
        assert!(!gpus[0].to_string().contains("compute capability"));
        gpus[0].cuda = Some(CudaInfo {
            ordinal: 0,
            compute_capability: ComputeCapability::new(6, 1),
        });
        assert!(
            gpus[0]
                .to_string()
                .ends_with("8060 MB VRAM, CUDA device 0 compute capability 6.1)")
        );
        assert_eq!(
            serde_json::to_value(gpus[0].cuda).unwrap(),
            serde_json::json!({"ordinal": 0, "compute_capability": "6.1"})
        );
    }

    #[test]
    fn platform_helpers() {
        let mac = HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]);
        assert!(mac.is_macos() && mac.is_apple_silicon() && !mac.is_windows());
        assert!(mac.has_vendor(GpuVendor::Apple) && !mac.has_vendor(GpuVendor::Intel));
        assert!(!HardwareInfo::new("macos", "x86_64", vec![]).is_apple_silicon());
        // Detection on the build host never panics and reports the host platform.
        let hw = hardware();
        assert_eq!(hw.os, std::env::consts::OS);
        assert_eq!(hw.arch, std::env::consts::ARCH);
    }
}
