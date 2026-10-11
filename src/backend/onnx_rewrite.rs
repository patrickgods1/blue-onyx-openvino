//! Graph rewrites that let the ONNX Runtime CoreML execution provider take a whole model.
//!
//! The CoreML EP (ONNX Runtime 1.24) has no `HardSigmoid`/`HardSwish` builder and accepts `Split`
//! only from opset 13 (`split` as an input) and `Flatten` not at all in MLProgram mode. YOLOv5
//! exports use Hardswish in every layer, so the graph falls apart into ~50 CoreML partitions with
//! CPU hops in between and runs slower than the plain CPU. Replacing those nodes by equivalent
//! elementwise ops CoreML supports makes it one partition:
//!
//! - `HardSigmoid(x; alpha, beta)` -> `Clip(Add(Mul(x, alpha), beta), 0, 1)`
//! - `HardSwish(x)` -> `Mul(x, Clip(Add(Mul(x, 1/6), 0.5), 0, 1))`
//! - `Split(x; axis, split)` -> one `Slice` per output (starts/ends/axes as int64 initializers)
//! - `Flatten(x; axis)` -> `Reshape(x, [0 | 1 | d0*..*d(axis-1), -1])` (opt-in: not part of
//!   [`RewriteOptions::COREML`], since on yolo26n it raised the partition count from 5 to 6
//!   without a speed-up; its other unsupported ops stay anyway)
//!
//! Clip takes min/max as inputs from opset 11 and as attributes before; Slice takes
//! starts/ends/axes as inputs from opset 10 and as attributes before.
//!
//! The model is handled at the protobuf wire-format level: every field the rewrite does not touch
//! (weights, doc strings, metadata, unknown fields, subgraphs) is copied byte for byte, only
//! `ModelProto.graph` and the replaced `GraphProto.node`s are re-encoded and new initializers are
//! appended. A model without anything to rewrite is reported as [`Outcome::Unchanged`]; models
//! with external data are left alone ([`Outcome::ExternalData`]), since a copy elsewhere would
//! lose the relative data paths. [`rewrite_cached`] writes the result next to other caches.

use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Bumped whenever the rewrite output changes, so cached copies are regenerated.
pub const REWRITER_VERSION: u32 = 1;

// ---- protobuf wire format ----------------------------------------------------------------------

/// A decoded field value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Value<'a> {
    Varint(u64),
    Fixed64(u64),
    Len(&'a [u8]),
    Fixed32(u32),
}

/// One field of a message: number, value and its complete encoding (key included), so untouched
/// fields can be written back unchanged.
#[derive(Debug, Clone, Copy)]
pub struct Field<'a> {
    pub num: u32,
    pub value: Value<'a>,
    pub raw: &'a [u8],
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let Some(&b) = buf.get(*pos) else {
            bail!("truncated varint at byte {pos}")
        };
        *pos += 1;
        v |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("varint longer than 10 bytes")
}

/// Split a message into its fields (groups, wire types 3/4, are not used by ONNX and rejected).
pub fn parse_fields(buf: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let start = pos;
        let key = read_varint(buf, &mut pos)?;
        let num = u32::try_from(key >> 3)
            .ok()
            .filter(|n| *n != 0)
            .with_context(|| format!("invalid field number at byte {start}"))?;
        let value = match key & 7 {
            0 => Value::Varint(read_varint(buf, &mut pos)?),
            1 => {
                let b = buf.get(pos..pos + 8).context("truncated fixed64")?;
                pos += 8;
                Value::Fixed64(u64::from_le_bytes(b.try_into().expect("8 bytes")))
            }
            2 => {
                let n = usize::try_from(read_varint(buf, &mut pos)?)?;
                let end = pos
                    .checked_add(n)
                    .filter(|e| *e <= buf.len())
                    .context("truncated length-delimited field")?;
                let v = &buf[pos..end];
                pos = end;
                Value::Len(v)
            }
            5 => {
                let b = buf.get(pos..pos + 4).context("truncated fixed32")?;
                pos += 4;
                Value::Fixed32(u32::from_le_bytes(b.try_into().expect("4 bytes")))
            }
            wt => bail!("unsupported wire type {wt} (field {num})"),
        };
        out.push(Field {
            num,
            value,
            raw: &buf[start..pos],
        });
    }
    Ok(out)
}

/// Protobuf encoder for the few field kinds the rewrite emits.
#[derive(Debug, Default)]
struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn key(&mut self, num: u32, wire_type: u64) {
        self.varint((u64::from(num) << 3) | wire_type);
    }
    fn bytes(&mut self, num: u32, b: &[u8]) {
        self.key(num, 2);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn string(&mut self, num: u32, s: &str) {
        self.bytes(num, s.as_bytes());
    }
    /// int32/int64/enum: negative values are sign-extended to 10 bytes, as protobuf does.
    fn int(&mut self, num: u32, v: i64) {
        self.key(num, 0);
        self.varint(v as u64);
    }
    fn float(&mut self, num: u32, v: f32) {
        self.key(num, 5);
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }
}

fn len_of<'a>(f: &Field<'a>) -> Option<&'a [u8]> {
    match f.value {
        Value::Len(b) => Some(b),
        _ => None,
    }
}

fn as_str(b: &[u8]) -> Result<&str> {
    std::str::from_utf8(b).context("string field is not UTF-8")
}

/// Last occurrence of a string field ("" when absent, as protobuf defaults).
fn get_str<'a>(fields: &[Field<'a>], num: u32) -> Result<&'a str> {
    match fields.iter().rev().find(|f| f.num == num).and_then(len_of) {
        Some(b) => as_str(b),
        None => Ok(""),
    }
}

fn get_len<'a>(fields: &[Field<'a>], num: u32) -> Option<&'a [u8]> {
    fields.iter().rev().find(|f| f.num == num).and_then(len_of)
}

fn get_int(fields: &[Field<'_>], num: u32) -> Option<i64> {
    fields
        .iter()
        .rev()
        .find_map(|f| match (f.num == num, f.value) {
            (true, Value::Varint(v)) => Some(v as i64),
            _ => None,
        })
}

fn get_float(fields: &[Field<'_>], num: u32) -> Option<f32> {
    fields
        .iter()
        .rev()
        .find_map(|f| match (f.num == num, f.value) {
            (true, Value::Fixed32(v)) => Some(f32::from_bits(v)),
            _ => None,
        })
}

fn rep_str<'a>(fields: &[Field<'a>], num: u32) -> Result<Vec<&'a str>> {
    fields
        .iter()
        .filter(|f| f.num == num)
        .filter_map(len_of)
        .map(as_str)
        .collect()
}

/// Repeated int32/int64, packed or not.
fn rep_int(fields: &[Field<'_>], num: u32) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for f in fields.iter().filter(|f| f.num == num) {
        match f.value {
            Value::Varint(v) => out.push(v as i64),
            Value::Len(b) => {
                let mut pos = 0;
                while pos < b.len() {
                    out.push(read_varint(b, &mut pos)? as i64);
                }
            }
            _ => bail!("field {num}: unexpected wire type for an integer"),
        }
    }
    Ok(out)
}

// ---- ONNX messages (only what the rewrite reads) -----------------------------------------------

// ModelProto
const MODEL_GRAPH: u32 = 7;
const MODEL_OPSET_IMPORT: u32 = 8;
const MODEL_METADATA_PROPS: u32 = 14;
// OperatorSetIdProto
const OPSET_DOMAIN: u32 = 1;
const OPSET_VERSION: u32 = 2;
// GraphProto
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_VALUE_INFO: u32 = 13;
// NodeProto
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_NAME: u32 = 3;
const NODE_OP_TYPE: u32 = 4;
const NODE_ATTRIBUTE: u32 = 5;
const NODE_DOMAIN: u32 = 7;
// AttributeProto
const ATTR_NAME: u32 = 1;
const ATTR_F: u32 = 2;
const ATTR_I: u32 = 3;
const ATTR_T: u32 = 5;
const ATTR_G: u32 = 6;
const ATTR_INTS: u32 = 8;
const ATTR_GRAPHS: u32 = 11;
const ATTR_TYPE: u32 = 20;
const ATTR_TYPE_FLOAT: i64 = 1;
const ATTR_TYPE_INTS: i64 = 7;
// TensorProto
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_INT32_DATA: u32 = 5;
const TENSOR_INT64_DATA: u32 = 7;
const TENSOR_NAME: u32 = 8;
const TENSOR_RAW_DATA: u32 = 9;
const TENSOR_EXTERNAL_DATA: u32 = 13;
const TENSOR_DATA_LOCATION: u32 = 14;
// ValueInfoProto / TypeProto / TensorShapeProto
const VI_NAME: u32 = 1;
const VI_TYPE: u32 = 2;
const TYPE_TENSOR: u32 = 1;
const TT_ELEM_TYPE: u32 = 1;
const TT_SHAPE: u32 = 2;
const SHAPE_DIM: u32 = 1;
const DIM_VALUE: u32 = 1;
// StringStringEntryProto
const ENTRY_KEY: u32 = 1;
const ENTRY_VALUE: u32 = 2;

// TensorProto.DataType
pub const DT_FLOAT: i32 = 1;
pub const DT_INT32: i32 = 6;
pub const DT_INT64: i32 = 7;
pub const DT_FLOAT16: i32 = 10;
pub const DT_DOUBLE: i32 = 11;

fn is_float_type(t: i32) -> bool {
    matches!(t, 1 | 10 | 11 | 16..=20 | 23)
}

#[derive(Debug)]
struct Attr<'a> {
    name: &'a str,
    f: Option<f32>,
    i: Option<i64>,
    ints: Vec<i64>,
    t: Option<&'a [u8]>,
    has_graph: bool,
}

fn parse_attr(b: &[u8]) -> Result<Attr<'_>> {
    let f = parse_fields(b)?;
    Ok(Attr {
        name: get_str(&f, ATTR_NAME)?,
        f: get_float(&f, ATTR_F),
        i: get_int(&f, ATTR_I),
        ints: rep_int(&f, ATTR_INTS)?,
        t: get_len(&f, ATTR_T),
        has_graph: f.iter().any(|x| x.num == ATTR_G || x.num == ATTR_GRAPHS),
    })
}

#[derive(Debug)]
struct Node<'a> {
    inputs: Vec<&'a str>,
    outputs: Vec<&'a str>,
    name: &'a str,
    op_type: &'a str,
    domain: &'a str,
    attrs: Vec<Attr<'a>>,
}

impl<'a> Node<'a> {
    fn parse(b: &'a [u8]) -> Result<Self> {
        let f = parse_fields(b)?;
        let attrs = f
            .iter()
            .filter(|x| x.num == NODE_ATTRIBUTE)
            .filter_map(len_of)
            .map(parse_attr)
            .collect::<Result<_>>()?;
        Ok(Self {
            inputs: rep_str(&f, NODE_INPUT)?,
            outputs: rep_str(&f, NODE_OUTPUT)?,
            name: get_str(&f, NODE_NAME)?,
            op_type: get_str(&f, NODE_OP_TYPE)?,
            domain: get_str(&f, NODE_DOMAIN)?,
            attrs,
        })
    }
    fn attr(&self, name: &str) -> Option<&Attr<'a>> {
        self.attrs.iter().rev().find(|a| a.name == name)
    }
    fn is_onnx(&self) -> bool {
        self.domain.is_empty() || self.domain == "ai.onnx"
    }
    /// Input `i` when present and not the empty "optional input omitted" name.
    fn input(&self, i: usize) -> Option<&'a str> {
        self.inputs.get(i).copied().filter(|s| !s.is_empty())
    }
}

#[derive(Debug)]
struct Tensor<'a> {
    name: &'a str,
    data_type: i32,
    dims: Vec<i64>,
    external: bool,
    fields: Vec<Field<'a>>,
}

impl<'a> Tensor<'a> {
    fn parse(b: &'a [u8]) -> Result<Self> {
        let fields = parse_fields(b)?;
        Ok(Self {
            name: get_str(&fields, TENSOR_NAME)?,
            data_type: get_int(&fields, TENSOR_DATA_TYPE).unwrap_or(0) as i32,
            dims: rep_int(&fields, TENSOR_DIMS)?,
            external: get_int(&fields, TENSOR_DATA_LOCATION) == Some(1)
                || fields.iter().any(|f| f.num == TENSOR_EXTERNAL_DATA),
            fields,
        })
    }

    /// Integer contents of an int64/int32 tensor (raw or typed data).
    fn ints(&self) -> Option<Vec<i64>> {
        if self.external {
            return None;
        }
        let raw = get_len(&self.fields, TENSOR_RAW_DATA);
        match self.data_type {
            DT_INT64 => match raw {
                Some(r) if r.len() % 8 == 0 => Some(
                    r.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| i64::from_le_bytes(*c))
                        .collect(),
                ),
                Some(_) => None,
                None => rep_int(&self.fields, TENSOR_INT64_DATA).ok(),
            },
            DT_INT32 => match raw {
                Some(r) if r.len() % 4 == 0 => Some(
                    r.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| i64::from(i32::from_le_bytes(*c)))
                        .collect(),
                ),
                Some(_) => None,
                None => rep_int(&self.fields, TENSOR_INT32_DATA)
                    .ok()
                    .map(|v| v.into_iter().map(|x| i64::from(x as i32)).collect()),
            },
            _ => None,
        }
    }
}

/// Element type and (partial) shape of a value from `ValueInfoProto`.
#[derive(Debug, Clone, Default, PartialEq)]
struct ValueType {
    elem: Option<i32>,
    /// None = rank unknown; `Some(None)` dims are symbolic/unknown.
    shape: Option<Vec<Option<i64>>>,
}

fn parse_value_info(b: &[u8]) -> Result<(&str, ValueType)> {
    let f = parse_fields(b)?;
    let name = get_str(&f, VI_NAME)?;
    let mut vt = ValueType::default();
    if let Some(tp) = get_len(&f, VI_TYPE) {
        let tp = parse_fields(tp)?;
        if let Some(tt) = get_len(&tp, TYPE_TENSOR) {
            let tt = parse_fields(tt)?;
            vt.elem = get_int(&tt, TT_ELEM_TYPE).map(|v| v as i32);
            if let Some(shape) = get_len(&tt, TT_SHAPE) {
                let shape = parse_fields(shape)?;
                let mut dims = Vec::new();
                for d in shape
                    .iter()
                    .filter(|x| x.num == SHAPE_DIM)
                    .filter_map(len_of)
                {
                    let d = parse_fields(d)?;
                    dims.push(get_int(&d, DIM_VALUE).filter(|v| *v >= 0));
                }
                vt.shape = Some(dims);
            }
        }
    }
    Ok((name, vt))
}

/// `ai.onnx` opset of a parsed ModelProto (None when the model imports none).
fn onnx_opset(model: &[Field<'_>]) -> Result<Option<i64>> {
    let mut opset = None;
    for b in model
        .iter()
        .filter(|f| f.num == MODEL_OPSET_IMPORT)
        .filter_map(len_of)
    {
        let f = parse_fields(b)?;
        let domain = get_str(&f, OPSET_DOMAIN)?;
        if domain.is_empty() || domain == "ai.onnx" {
            opset = get_int(&f, OPSET_VERSION);
        }
    }
    Ok(opset)
}

// ---- options and results -----------------------------------------------------------------------

/// Which `Split` nodes to replace by `Slice`s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitRewrite {
    Never,
    /// Only below opset 13 (sizes as the `split` attribute), which the CoreML EP rejects.
    BelowOpset13,
    Always,
}

/// Which rewrites to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewriteOptions {
    pub hard_sigmoid: bool,
    pub hard_swish: bool,
    pub split: SplitRewrite,
    pub flatten: bool,
}

impl RewriteOptions {
    /// What the ONNX Runtime 1.24 CoreML EP (MLProgram) cannot take but supports once rewritten.
    /// Measured on YOLOv5 (IPcam-*, delivery, package, ipcam-bird): one CoreML partition instead
    /// of 2-53, outputs bit-identical on the CPU EP.
    pub const COREML: Self = Self {
        hard_sigmoid: true,
        hard_swish: true,
        split: SplitRewrite::BelowOpset13,
        flatten: false,
    };
    /// Every rewrite (tests).
    pub const ALL: Self = Self {
        hard_sigmoid: true,
        hard_swish: true,
        split: SplitRewrite::Always,
        flatten: true,
    };
}

/// A rewritten model.
#[derive(Debug, Clone)]
pub struct Rewrite {
    pub bytes: Vec<u8>,
    /// Rewritten nodes per op type.
    pub counts: BTreeMap<String, usize>,
    /// Nodes that matched but could not be rewritten, with the reason.
    pub skipped: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    /// Nothing to rewrite (`skipped`: candidates that could not be rewritten).
    Unchanged {
        skipped: Vec<String>,
    },
    /// The model keeps its weights in external files; left alone.
    ExternalData,
    Rewritten(Rewrite),
}

/// "HardSigmoid x51, Split x3".
pub fn summary(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .map(|(k, v)| format!("{k} x{v}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---- rewrite -----------------------------------------------------------------------------------

/// Everything known about the main graph.
struct Graph<'a> {
    fields: Vec<Field<'a>>,
    /// (field index, node)
    nodes: Vec<(usize, Node<'a>)>,
    /// name -> (field index, tensor)
    inits: HashMap<&'a str, (usize, Tensor<'a>)>,
    types: HashMap<&'a str, ValueType>,
    inputs: HashSet<&'a str>,
    outputs: HashSet<&'a str>,
    names: HashSet<String>,
    has_subgraphs: bool,
    external_data: bool,
    /// Element type assumed for float values whose type is not recorded: f32 when no other
    /// floating type occurs anywhere in the model.
    default_float: Option<i32>,
}

impl<'a> Graph<'a> {
    fn parse(b: &'a [u8]) -> Result<Self> {
        let fields = parse_fields(b)?;
        let mut g = Graph {
            fields: Vec::new(),
            nodes: Vec::new(),
            inits: HashMap::new(),
            types: HashMap::new(),
            inputs: HashSet::new(),
            outputs: HashSet::new(),
            names: HashSet::new(),
            has_subgraphs: false,
            external_data: false,
            default_float: None,
        };
        let mut float_types = BTreeSet::new();
        for (i, f) in fields.iter().enumerate() {
            let Some(b) = len_of(f) else { continue };
            match f.num {
                GRAPH_NODE => {
                    let n = Node::parse(b).with_context(|| format!("node #{}", g.nodes.len()))?;
                    for a in &n.attrs {
                        g.has_subgraphs |= a.has_graph;
                        if let Some(t) = a.t {
                            let t = Tensor::parse(t)?;
                            g.external_data |= t.external;
                            float_types.insert(t.data_type);
                        }
                    }
                    if matches!(n.op_type, "Cast" | "CastLike")
                        && let Some(to) = n.attr("to").and_then(|a| a.i)
                    {
                        float_types.insert(to as i32);
                    }
                    g.names.extend(n.inputs.iter().map(|s| s.to_string()));
                    g.names.extend(n.outputs.iter().map(|s| s.to_string()));
                    g.names.insert(n.name.to_string());
                    g.nodes.push((i, n));
                }
                GRAPH_INITIALIZER => {
                    let t = Tensor::parse(b)?;
                    g.external_data |= t.external;
                    float_types.insert(t.data_type);
                    g.names.insert(t.name.to_string());
                    g.inits.insert(t.name, (i, t));
                }
                GRAPH_INPUT | GRAPH_OUTPUT | GRAPH_VALUE_INFO => {
                    let (name, vt) = parse_value_info(b)?;
                    if let Some(e) = vt.elem {
                        float_types.insert(e);
                    }
                    g.names.insert(name.to_string());
                    match f.num {
                        GRAPH_INPUT => g.inputs.insert(name),
                        GRAPH_OUTPUT => g.outputs.insert(name),
                        _ => false,
                    };
                    g.types.entry(name).or_insert(vt);
                }
                _ => {}
            }
        }
        float_types.retain(|t| is_float_type(*t));
        g.default_float = match float_types.iter().collect::<Vec<_>>()[..] {
            [] | [&DT_FLOAT] => Some(DT_FLOAT),
            _ => None,
        };
        g.fields = fields;
        Ok(g)
    }

    fn elem_type(&self, name: &str) -> Option<i32> {
        self.types
            .get(name)
            .and_then(|t| t.elem)
            .or_else(|| self.inits.get(name).map(|(_, t)| t.data_type))
            .or(self.default_float)
    }

    fn shape(&self, name: &str) -> Option<Vec<Option<i64>>> {
        self.types
            .get(name)
            .and_then(|t| t.shape.clone())
            .or_else(|| {
                self.inits
                    .get(name)
                    .map(|(_, t)| t.dims.iter().map(|d| Some(*d)).collect())
            })
    }

    /// Integer contents of an initializer or `Constant` output.
    fn const_ints(&self, name: &str) -> Option<Vec<i64>> {
        if let Some((_, t)) = self.inits.get(name) {
            return t.ints();
        }
        let (_, n) = self
            .nodes
            .iter()
            .find(|(_, n)| n.op_type == "Constant" && n.outputs.first() == Some(&name))?;
        if let Some(a) = n.attr("value_ints") {
            return Some(a.ints.clone());
        }
        if let Some(a) = n.attr("value_int") {
            return a.i.map(|v| vec![v]);
        }
        Tensor::parse(n.attr("value")?.t?).ok()?.ints()
    }
}

/// Encodes new nodes and initializers with names unique in the graph.
struct Builder<'g> {
    names: &'g mut HashSet<String>,
    inits: Vec<Vec<u8>>,
    consts: HashMap<(i32, Vec<i64>, Vec<u8>), String>,
}

/// Attribute values the rewrite emits.
enum AttrVal {
    F(f32),
    Ints(Vec<i64>),
}

impl Builder<'_> {
    fn fresh(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut k = 1;
        while self.names.contains(&name) {
            name = format!("{base}_{k}");
            k += 1;
        }
        self.names.insert(name.clone());
        name
    }

    /// A shared initializer (one per distinct type/shape/value).
    fn constant(&mut self, data_type: i32, dims: &[i64], raw: Vec<u8>) -> String {
        let key = (data_type, dims.to_vec(), raw);
        if let Some(n) = self.consts.get(&key) {
            return n.clone();
        }
        let name = self.fresh(&format!("bop_rewrite/const_{}", self.consts.len()));
        let mut w = Writer::default();
        for d in dims {
            w.int(TENSOR_DIMS, *d);
        }
        w.int(TENSOR_DATA_TYPE, i64::from(data_type));
        w.string(TENSOR_NAME, &name);
        w.bytes(TENSOR_RAW_DATA, &key.2);
        self.inits.push(w.buf);
        self.consts.insert(key, name.clone());
        name
    }

    /// Scalar of a floating element type (f32, f16, f64); None for other types.
    fn scalar(&mut self, data_type: i32, v: f32) -> Option<String> {
        let raw = match data_type {
            DT_FLOAT => v.to_le_bytes().to_vec(),
            DT_DOUBLE => f64::from(v).to_le_bytes().to_vec(),
            DT_FLOAT16 => f32_to_f16_bits(v).to_le_bytes().to_vec(),
            _ => return None,
        };
        Some(self.constant(data_type, &[], raw))
    }

    fn ints(&mut self, v: &[i64]) -> String {
        let raw = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.constant(DT_INT64, &[v.len() as i64], raw)
    }
}

fn encode_node(
    inputs: &[&str],
    outputs: &[&str],
    name: &str,
    op: &str,
    attrs: &[(&str, AttrVal)],
) -> Vec<u8> {
    let mut w = Writer::default();
    for i in inputs {
        w.string(NODE_INPUT, i);
    }
    for o in outputs {
        w.string(NODE_OUTPUT, o);
    }
    w.string(NODE_NAME, name);
    w.string(NODE_OP_TYPE, op);
    for (n, v) in attrs {
        let mut a = Writer::default();
        a.string(ATTR_NAME, n);
        match v {
            AttrVal::F(f) => {
                a.float(ATTR_F, *f);
                a.int(ATTR_TYPE, ATTR_TYPE_FLOAT);
            }
            AttrVal::Ints(v) => {
                for x in v {
                    a.int(ATTR_INTS, *x);
                }
                a.int(ATTR_TYPE, ATTR_TYPE_INTS);
            }
        }
        w.bytes(NODE_ATTRIBUTE, &a.buf);
    }
    w.buf
}

/// IEEE 754 binary16 bits of `x`, rounded to nearest even.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1; // may carry into the exponent (up to infinity), which is the correct rounding
    }
    sign | r as u16
}

/// Per-node context for emitting replacement nodes.
struct Emit<'b, 'g> {
    b: &'b mut Builder<'g>,
    base: String,
    nodes: Vec<Vec<u8>>,
}

impl Emit<'_, '_> {
    fn node(&mut self, inputs: &[&str], outputs: &[&str], suffix: &str, op: &str) {
        self.node_with(inputs, outputs, suffix, op, &[]);
    }
    fn node_with(
        &mut self,
        inputs: &[&str],
        outputs: &[&str],
        suffix: &str,
        op: &str,
        attrs: &[(&str, AttrVal)],
    ) {
        let name = self.b.fresh(&format!("{}/bop_{suffix}", self.base));
        self.nodes
            .push(encode_node(inputs, outputs, &name, op, attrs));
    }
}

/// `y = Clip(Add(Mul(x, alpha), beta), 0, 1)`; HardSwish multiplies by `x` on top.
fn emit_hard_sigmoid(
    e: &mut Emit<'_, '_>,
    opset: i64,
    dtype: i32,
    x: &str,
    (alpha, beta): (f32, f32),
    y: &str,
    swish: bool,
) -> Result<(), String> {
    let (Some(a), Some(b)) = (e.b.scalar(dtype, alpha), e.b.scalar(dtype, beta)) else {
        return Err(format!("element type {dtype} not supported"));
    };
    let mul = e.b.fresh(&format!("{y}/bop_mul"));
    let add = e.b.fresh(&format!("{y}/bop_add"));
    let clip = if swish {
        e.b.fresh(&format!("{y}/bop_clip"))
    } else {
        y.to_string()
    };
    e.node(&[x, &a], &[&mul], "mul", "Mul");
    e.node(&[&mul, &b], &[&add], "add", "Add");
    if opset >= 11 {
        let lo = e.b.scalar(dtype, 0.0).expect("type checked above");
        let hi = e.b.scalar(dtype, 1.0).expect("type checked above");
        e.node(&[&add, &lo, &hi], &[&clip], "clip", "Clip");
    } else {
        e.node_with(
            &[&add],
            &[&clip],
            "clip",
            "Clip",
            &[("min", AttrVal::F(0.0)), ("max", AttrVal::F(1.0))],
        );
    }
    if swish {
        e.node(&[x, &clip], &[y], "mul_x", "Mul");
    }
    Ok(())
}

/// Split sizes along `axis` of a `Split` node, or why they are unknown.
fn split_sizes(g: &Graph<'_>, n: &Node<'_>, opset: i64, axis: i64) -> Result<Vec<i64>, String> {
    let outs = n.outputs.len() as i64;
    let explicit = if opset < 13 {
        n.attr("split").map(|a| a.ints.clone())
    } else if let Some(s) = n.input(1) {
        Some(
            g.const_ints(s)
                .ok_or_else(|| format!("split sizes '{s}' are not a constant"))?,
        )
    } else {
        None
    };
    if let Some(v) = explicit {
        return Ok(v);
    }
    let x = n.inputs.first().copied().unwrap_or_default();
    let dim = g
        .shape(x)
        .and_then(|s| {
            let r = s.len() as i64;
            let a = if axis < 0 { axis + r } else { axis };
            s.get(usize::try_from(a).ok()?).copied().flatten()
        })
        .ok_or_else(|| format!("equal split of '{x}' needs its shape along axis {axis}"))?;
    if opset >= 18 {
        let k = n
            .attr("num_outputs")
            .and_then(|a| a.i)
            .ok_or("neither 'split' nor 'num_outputs' given")?;
        if k <= 0 || k != outs {
            return Err(format!("num_outputs {k} != {outs} outputs"));
        }
        // Chunks of ceil(dim / k); the last one is smaller when uneven.
        let chunk = (dim + k - 1) / k;
        let mut v = Vec::new();
        let mut left = dim;
        for _ in 0..k {
            v.push(chunk.min(left).max(0));
            left -= chunk;
        }
        return Ok(v);
    }
    if outs == 0 || dim % outs != 0 {
        return Err(format!("dimension {dim} does not split evenly into {outs}"));
    }
    Ok(vec![dim / outs; outs as usize])
}

fn emit_split(e: &mut Emit<'_, '_>, g: &Graph<'_>, n: &Node<'_>, opset: i64) -> Result<(), String> {
    let x = n.input(0).ok_or("no input")?;
    let axis = n.attr("axis").and_then(|a| a.i).unwrap_or(0);
    let sizes = split_sizes(g, n, opset, axis)?;
    if sizes.len() != n.outputs.len() || sizes.iter().any(|s| *s < 0) {
        return Err(format!(
            "{} split sizes {sizes:?} for {} outputs",
            sizes.len(),
            n.outputs.len()
        ));
    }
    let mut start = 0i64;
    for (j, (size, out)) in sizes.iter().zip(&n.outputs).enumerate() {
        let end = start + size;
        if !out.is_empty() {
            let suffix = format!("slice{j}");
            if opset >= 10 {
                let s = e.b.ints(&[start]);
                let en = e.b.ints(&[end]);
                let ax = e.b.ints(&[axis]);
                e.node(&[x, &s, &en, &ax], &[out], &suffix, "Slice");
            } else {
                e.node_with(
                    &[x],
                    &[out],
                    &suffix,
                    "Slice",
                    &[
                        ("starts", AttrVal::Ints(vec![start])),
                        ("ends", AttrVal::Ints(vec![end])),
                        ("axes", AttrVal::Ints(vec![axis])),
                    ],
                );
            }
        }
        start = end;
    }
    Ok(())
}

fn emit_flatten(e: &mut Emit<'_, '_>, g: &Graph<'_>, n: &Node<'_>) -> Result<(), String> {
    let x = n.input(0).ok_or("no input")?;
    let y = n.outputs.first().copied().ok_or("no output")?;
    let mut axis = n.attr("axis").and_then(|a| a.i).unwrap_or(1);
    let shape = g.shape(x);
    if axis < 0 {
        let r = shape
            .as_ref()
            .ok_or("negative axis needs the input rank")?
            .len() as i64;
        axis += r;
    }
    let target = match axis {
        0 => vec![1, -1],
        // Reshape copies dimension 0 for a 0 entry (allowzero = 0).
        1 => vec![0, -1],
        a if a > 1 => {
            let dims = shape.ok_or("axis > 1 needs the input shape")?;
            let lead: Option<Vec<i64>> = dims
                .get(..a as usize)
                .ok_or("axis beyond the input rank")?
                .iter()
                .copied()
                .collect();
            let lead = lead.ok_or("axis > 1 needs static leading dimensions")?;
            vec![lead.iter().product(), -1]
        }
        _ => return Err(format!("axis {axis} out of range")),
    };
    let s = e.b.ints(&target);
    e.node(&[x, &s], &[y], "reshape", "Reshape");
    Ok(())
}

/// Rewrite `model` (serialized ModelProto) as `opts` asks.
pub fn rewrite(model: &[u8], opts: &RewriteOptions) -> Result<Outcome> {
    let mfields = parse_fields(model).context("parsing ModelProto")?;
    let Some(gi) = mfields.iter().rposition(|f| f.num == MODEL_GRAPH) else {
        bail!("model has no graph");
    };
    let gbytes = len_of(&mfields[gi]).context("graph field is not length-delimited")?;
    let opset = onnx_opset(&mfields)?;
    let mut g = Graph::parse(gbytes).context("parsing GraphProto")?;
    if g.external_data {
        return Ok(Outcome::ExternalData);
    }
    let mut skipped = Vec::new();
    let Some(opset) = opset else {
        return Ok(Outcome::Unchanged { skipped });
    };

    let mut names = std::mem::take(&mut g.names);
    let mut b = Builder {
        names: &mut names,
        inits: Vec::new(),
        consts: HashMap::new(),
    };
    // field index of a replaced node -> replacement nodes
    let mut replaced: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut maybe_dead: Vec<&str> = Vec::new();
    for (fi, n) in &g.nodes {
        if !n.is_onnx() {
            continue;
        }
        let wanted = match n.op_type {
            "HardSigmoid" => opts.hard_sigmoid,
            "HardSwish" => opts.hard_swish,
            "Split" => match opts.split {
                SplitRewrite::Never => false,
                SplitRewrite::BelowOpset13 => opset < 13,
                SplitRewrite::Always => true,
            },
            "Flatten" => opts.flatten,
            _ => false,
        };
        if !wanted {
            continue;
        }
        let base = if n.name.is_empty() {
            format!("bop_rewrite/{}", n.op_type)
        } else {
            n.name.to_string()
        };
        let mut e = Emit {
            b: &mut b,
            base,
            nodes: Vec::new(),
        };
        let r = match n.op_type {
            "HardSigmoid" | "HardSwish" => {
                let swish = n.op_type == "HardSwish";
                match (n.input(0), n.outputs.first()) {
                    (Some(x), Some(y)) if opset >= 7 => match g.elem_type(x) {
                        Some(dt) => {
                            let (alpha, beta) = if swish {
                                (1.0 / 6.0, 0.5)
                            } else {
                                (
                                    n.attr("alpha").and_then(|a| a.f).unwrap_or(0.2),
                                    n.attr("beta").and_then(|a| a.f).unwrap_or(0.5),
                                )
                            };
                            emit_hard_sigmoid(&mut e, opset, dt, x, (alpha, beta), y, swish)
                        }
                        None => Err(format!("element type of '{x}' unknown")),
                    },
                    _ if opset < 7 => Err(format!("opset {opset} < 7")),
                    _ => Err("missing input or output".to_string()),
                }
            }
            "Split" => emit_split(&mut e, &g, n, opset),
            "Flatten" if opset >= 5 => emit_flatten(&mut e, &g, n),
            "Flatten" => Err(format!("opset {opset} < 5")),
            _ => unreachable!("filtered above"),
        };
        match r {
            Ok(()) => {
                *counts.entry(n.op_type.to_string()).or_default() += 1;
                replaced.insert(*fi, e.nodes);
                if n.op_type == "Split"
                    && opset >= 13
                    && let Some(s) = n.input(1)
                {
                    maybe_dead.push(s);
                }
            }
            Err(why) => {
                // Names minted for a failed attempt stay reserved; harmless.
                let label = if n.name.is_empty() {
                    "<unnamed>"
                } else {
                    n.name
                };
                skipped.push(format!("{} '{label}': {why}", n.op_type));
            }
        }
    }
    if replaced.is_empty() {
        return Ok(Outcome::Unchanged { skipped });
    }
    let new_inits = std::mem::take(&mut b.inits);

    // Split sizes nobody reads any more: drop their initializer / Constant node (ONNX Runtime
    // warns about unused initializers). Not attempted when subgraphs could reference them.
    let mut dropped: HashSet<usize> = HashSet::new();
    if !g.has_subgraphs {
        let used: HashSet<&str> = g
            .nodes
            .iter()
            .filter(|(fi, _)| !replaced.contains_key(fi))
            .flat_map(|(_, n)| n.inputs.iter().copied())
            .chain(g.outputs.iter().copied())
            .chain(g.inputs.iter().copied())
            .collect();
        for name in maybe_dead {
            if used.contains(name) {
                continue;
            }
            if let Some((fi, _)) = g.inits.get(name) {
                dropped.insert(*fi);
            } else if let Some((fi, _)) = g.nodes.iter().find(|(_, n)| {
                n.op_type == "Constant" && n.outputs.len() == 1 && n.outputs[0] == name
            }) {
                dropped.insert(*fi);
            }
        }
    }

    let marker = format!("v{REWRITER_VERSION}: {}", summary(&counts));
    let bytes = assemble(
        &mfields,
        gi,
        &g.fields,
        &replaced,
        &dropped,
        &new_inits,
        Some(&marker),
    );
    Ok(Outcome::Rewritten(Rewrite {
        bytes,
        counts,
        skipped,
    }))
}

/// Serialize the model again: untouched fields byte for byte, `replaced` graph fields (by index)
/// by their new nodes, `dropped` graph fields left out, `new_inits` appended to the graph and a
/// `marker` metadata entry appended to the model.
fn assemble(
    mfields: &[Field<'_>],
    graph_index: usize,
    gfields: &[Field<'_>],
    replaced: &HashMap<usize, Vec<Vec<u8>>>,
    dropped: &HashSet<usize>,
    new_inits: &[Vec<u8>],
    marker: Option<&str>,
) -> Vec<u8> {
    let mut gw = Writer::default();
    for (i, f) in gfields.iter().enumerate() {
        if dropped.contains(&i) {
            continue;
        }
        match replaced.get(&i) {
            Some(nodes) => {
                for n in nodes {
                    gw.bytes(GRAPH_NODE, n);
                }
            }
            None => gw.raw(f.raw),
        }
    }
    for t in new_inits {
        gw.bytes(GRAPH_INITIALIZER, t);
    }
    let mut mw = Writer::default();
    for (i, f) in mfields.iter().enumerate() {
        if i == graph_index {
            mw.bytes(MODEL_GRAPH, &gw.buf);
        } else {
            mw.raw(f.raw);
        }
    }
    if let Some(m) = marker {
        let mut entry = Writer::default();
        entry.string(ENTRY_KEY, "blue_onyx_prism.rewrite");
        entry.string(ENTRY_VALUE, m);
        mw.bytes(MODEL_METADATA_PROPS, &entry.buf);
    }
    mw.buf
}

/// Parse `model` and serialize it again without changes; byte-identical to the input for any
/// model with minimal varints (what every protobuf encoder writes). Checks the wire-format layer.
pub fn reencode(model: &[u8]) -> Result<Vec<u8>> {
    let mfields = parse_fields(model)?;
    let gi = mfields
        .iter()
        .rposition(|f| f.num == MODEL_GRAPH)
        .context("model has no graph")?;
    let gfields = parse_fields(len_of(&mfields[gi]).context("graph field")?)?;
    Ok(assemble(
        &mfields,
        gi,
        &gfields,
        &HashMap::new(),
        &HashSet::new(),
        &[],
        None,
    ))
}

// ---- cached copies -----------------------------------------------------------------------------

/// Result of [`rewrite_cached`].
#[derive(Debug, Clone)]
pub enum Cached {
    /// Use the original model (`skipped`: candidates that could not be rewritten).
    Original { skipped: Vec<String> },
    /// The model has external data; use the original.
    ExternalData,
    /// Load `path` instead. `created` is false when an earlier copy was reused (then `counts`
    /// are recomputed all the same, so logs stay informative).
    Rewritten {
        path: PathBuf,
        counts: BTreeMap<String, usize>,
        skipped: Vec<String>,
        created: bool,
    },
}

/// Cache file name for a rewrite of `src` with contents hash `sha_hex`.
pub fn cache_file_name(src: &Path, sha_hex: &str) -> String {
    let stem = src
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let short = &sha_hex[..sha_hex.len().min(16)];
    format!("{stem}-{short}-v{REWRITER_VERSION}.onnx")
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Rewrite `src` into `dir/<stem>-<sha256 16 hex>-v<version>.onnx` (written to a temporary file
/// and renamed into place; reused when it exists). `src` itself is never modified.
pub fn rewrite_cached(src: &Path, dir: &Path, opts: &RewriteOptions) -> Result<Cached> {
    let bytes = std::fs::read(src).with_context(|| format!("reading {}", src.display()))?;
    let r = match rewrite(&bytes, opts).with_context(|| format!("parsing {}", src.display()))? {
        Outcome::Unchanged { skipped } => return Ok(Cached::Original { skipped }),
        Outcome::ExternalData => return Ok(Cached::ExternalData),
        Outcome::Rewritten(r) => r,
    };
    let path = dir.join(cache_file_name(src, &sha256_hex(&bytes)));
    drop(bytes);
    if std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() > 0) {
        return Ok(Cached::Rewritten {
            path,
            counts: r.counts,
            skipped: r.skipped,
            created: false,
        });
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{}.tmp-{}-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let write = || -> Result<()> {
        std::fs::write(&tmp, &r.bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(Cached::Rewritten {
        path,
        counts: r.counts,
        skipped: r.skipped,
        created: true,
    })
}

// ---- tests -------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_round_trip() {
        for v in [0u64, 1, 127, 128, 300, 1 << 35, u64::MAX] {
            let mut w = Writer::default();
            w.varint(v);
            let mut pos = 0;
            assert_eq!(read_varint(&w.buf, &mut pos).unwrap(), v);
            assert_eq!(pos, w.buf.len());
        }
        assert!(read_varint(&[0x80], &mut 0).is_err());
        assert!(read_varint(&[0xff; 11], &mut 0).is_err());
    }

    #[test]
    fn negative_ints_are_ten_bytes() {
        let mut w = Writer::default();
        w.int(3, -1);
        assert_eq!(w.buf.len(), 11);
        let f = parse_fields(&w.buf).unwrap();
        assert_eq!(get_int(&f, 3), Some(-1));
    }

    #[test]
    fn truncated_messages_are_errors() {
        assert!(parse_fields(&[0x0a, 0x05, 1, 2]).is_err());
        assert!(parse_fields(&[0x0d, 1, 2]).is_err());
        assert!(parse_fields(&[0x0b]).is_err(), "groups are rejected");
        assert!(parse_fields(&[0x00, 0x00]).is_err(), "field number 0");
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(1.0 / 6.0), 0x3155);
        assert_eq!(f32_to_f16_bits(0.2), 0x3266);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00, "rounds up to infinity");
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::NAN) & 0x7e00, 0x7e00);
        assert_eq!(
            f32_to_f16_bits(2f32.powi(-24)),
            0x0001,
            "smallest subnormal"
        );
        assert_eq!(f32_to_f16_bits(2f32.powi(-14)), 0x0400, "smallest normal");
        assert_eq!(f32_to_f16_bits(2f32.powi(-26)), 0x0000);
        // 1 + 2^-11 is halfway between 1 and the next f16: ties to even (1.0).
        assert_eq!(f32_to_f16_bits(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
    }

    #[test]
    fn summary_lists_counts() {
        let mut c = BTreeMap::new();
        c.insert("Split".to_string(), 3);
        c.insert("HardSigmoid".to_string(), 51);
        assert_eq!(summary(&c), "HardSigmoid x51, Split x3");
    }

    // -- synthetic ONNX models -------------------------------------------------------------

    fn msg(f: impl FnOnce(&mut Writer)) -> Vec<u8> {
        let mut w = Writer::default();
        f(&mut w);
        w.buf
    }

    fn attr_f(name: &str, v: f32) -> Vec<u8> {
        msg(|w| {
            w.string(ATTR_NAME, name);
            w.float(ATTR_F, v);
            w.int(ATTR_TYPE, ATTR_TYPE_FLOAT);
        })
    }

    fn attr_i(name: &str, v: i64) -> Vec<u8> {
        msg(|w| {
            w.string(ATTR_NAME, name);
            w.int(ATTR_I, v);
            w.int(ATTR_TYPE, 2);
        })
    }

    /// `packed`: encode like proto3/packed writers do (one length-delimited field).
    fn attr_ints(name: &str, v: &[i64], packed: bool) -> Vec<u8> {
        msg(|w| {
            w.string(ATTR_NAME, name);
            if packed {
                let p = msg(|p| v.iter().for_each(|x| p.varint(*x as u64)));
                w.bytes(ATTR_INTS, &p);
            } else {
                v.iter().for_each(|x| w.int(ATTR_INTS, *x));
            }
            w.int(ATTR_TYPE, ATTR_TYPE_INTS);
        })
    }

    fn attr_t(name: &str, t: Vec<u8>) -> Vec<u8> {
        msg(|w| {
            w.string(ATTR_NAME, name);
            w.bytes(ATTR_T, &t);
            w.int(ATTR_TYPE, 4);
        })
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
        msg(|w| {
            inputs.iter().for_each(|i| w.string(NODE_INPUT, i));
            outputs.iter().for_each(|o| w.string(NODE_OUTPUT, o));
            w.string(NODE_NAME, name);
            w.string(NODE_OP_TYPE, op);
            attrs.iter().for_each(|a| w.bytes(NODE_ATTRIBUTE, a));
        })
    }

    fn tensor_i64(name: &str, vals: &[i64], raw: bool) -> Vec<u8> {
        msg(|w| {
            w.int(TENSOR_DIMS, vals.len() as i64);
            w.int(TENSOR_DATA_TYPE, i64::from(DT_INT64));
            if !raw {
                let p = msg(|p| vals.iter().for_each(|x| p.varint(*x as u64)));
                w.bytes(TENSOR_INT64_DATA, &p);
            }
            w.string(TENSOR_NAME, name);
            if raw {
                let r: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
                w.bytes(TENSOR_RAW_DATA, &r);
            }
        })
    }

    fn tensor_f32(name: &str, dims: &[i64], vals: &[f32], dtype: i32) -> Vec<u8> {
        msg(|w| {
            dims.iter().for_each(|d| w.int(TENSOR_DIMS, *d));
            w.int(TENSOR_DATA_TYPE, i64::from(dtype));
            w.string(TENSOR_NAME, name);
            let r: Vec<u8> = vals.iter().flat_map(|x| x.to_le_bytes()).collect();
            w.bytes(TENSOR_RAW_DATA, &r);
        })
    }

    /// `dims`: None = no shape; Some(d) with -1 for a symbolic dimension.
    fn value_info(name: &str, elem: i32, dims: Option<&[i64]>) -> Vec<u8> {
        let tt = msg(|w| {
            w.int(TT_ELEM_TYPE, i64::from(elem));
            if let Some(d) = dims {
                let shape = msg(|s| {
                    for x in d {
                        let dim = msg(|dw| {
                            if *x >= 0 {
                                dw.int(DIM_VALUE, *x);
                            } else {
                                dw.string(2, "N");
                            }
                        });
                        s.bytes(SHAPE_DIM, &dim);
                    }
                });
                w.bytes(TT_SHAPE, &shape);
            }
        });
        let tp = msg(|w| w.bytes(TYPE_TENSOR, &tt));
        msg(|w| {
            w.string(VI_NAME, name);
            w.bytes(VI_TYPE, &tp);
        })
    }

    #[derive(Default)]
    struct G {
        nodes: Vec<Vec<u8>>,
        inits: Vec<Vec<u8>>,
        inputs: Vec<Vec<u8>>,
        outputs: Vec<Vec<u8>>,
        value_info: Vec<Vec<u8>>,
        /// raw extra graph fields (unknown fields)
        extra: Vec<u8>,
    }

    fn model(opset: i64, g: &G, extra_model: &[u8]) -> Vec<u8> {
        let graph = msg(|w| {
            g.nodes.iter().for_each(|n| w.bytes(GRAPH_NODE, n));
            w.string(2, "test");
            g.inits.iter().for_each(|t| w.bytes(GRAPH_INITIALIZER, t));
            g.inputs.iter().for_each(|v| w.bytes(GRAPH_INPUT, v));
            g.outputs.iter().for_each(|v| w.bytes(GRAPH_OUTPUT, v));
            g.value_info
                .iter()
                .for_each(|v| w.bytes(GRAPH_VALUE_INFO, v));
            w.raw(&g.extra);
        });
        let opset_id = msg(|w| {
            w.string(OPSET_DOMAIN, "");
            w.int(OPSET_VERSION, opset);
        });
        msg(|w| {
            w.int(1, 8);
            w.string(2, "bop-test");
            w.bytes(MODEL_GRAPH, &graph);
            w.bytes(MODEL_OPSET_IMPORT, &opset_id);
            w.raw(extra_model);
        })
    }

    /// Single-node model `x -> op -> y` with f32 input/output of `dims`.
    fn single(opset: i64, op_node: Vec<u8>, outputs: &[&str], dims: &[i64]) -> Vec<u8> {
        let g = G {
            nodes: vec![op_node],
            inputs: vec![value_info("x", DT_FLOAT, Some(dims))],
            outputs: outputs
                .iter()
                .map(|o| value_info(o, DT_FLOAT, None))
                .collect(),
            ..G::default()
        };
        model(opset, &g, &[])
    }

    /// (op_type, inputs, outputs, attributes as (name, f, ints))
    type ViewNode = (
        String,
        Vec<String>,
        Vec<String>,
        Vec<(String, Option<f32>, Vec<i64>)>,
    );

    /// Decoded view of a rewritten model for assertions.
    struct View {
        nodes: Vec<ViewNode>,
        /// name -> (data_type, dims, raw)
        inits: HashMap<String, (i32, Vec<i64>, Vec<u8>)>,
        metadata: Vec<(String, String)>,
    }

    fn view(bytes: &[u8]) -> View {
        let m = parse_fields(bytes).unwrap();
        let g = parse_fields(get_len(&m, MODEL_GRAPH).unwrap()).unwrap();
        let mut v = View {
            nodes: Vec::new(),
            inits: HashMap::new(),
            metadata: Vec::new(),
        };
        for f in &g {
            let b = len_of(f).unwrap_or_default();
            match f.num {
                GRAPH_NODE => {
                    let n = Node::parse(b).unwrap();
                    v.nodes.push((
                        n.op_type.to_string(),
                        n.inputs.iter().map(|s| s.to_string()).collect(),
                        n.outputs.iter().map(|s| s.to_string()).collect(),
                        n.attrs
                            .iter()
                            .map(|a| (a.name.to_string(), a.f, a.ints.clone()))
                            .collect(),
                    ));
                }
                GRAPH_INITIALIZER => {
                    let t = Tensor::parse(b).unwrap();
                    let raw = get_len(&t.fields, TENSOR_RAW_DATA)
                        .unwrap_or_default()
                        .to_vec();
                    v.inits
                        .insert(t.name.to_string(), (t.data_type, t.dims.clone(), raw));
                }
                _ => {}
            }
        }
        for b in m.iter().filter(|f| f.num == MODEL_METADATA_PROPS) {
            let e = parse_fields(len_of(b).unwrap()).unwrap();
            v.metadata.push((
                get_str(&e, ENTRY_KEY).unwrap().into(),
                get_str(&e, ENTRY_VALUE).unwrap().into(),
            ));
        }
        v
    }

    impl View {
        fn ops(&self) -> Vec<&str> {
            self.nodes.iter().map(|n| n.0.as_str()).collect()
        }
        fn f32_of(&self, name: &str) -> f32 {
            let (dt, dims, raw) = &self.inits[name];
            assert_eq!((*dt, dims.len()), (DT_FLOAT, 0), "{name}: f32 scalar");
            f32::from_le_bytes(raw[..].try_into().unwrap())
        }
        fn i64s_of(&self, name: &str) -> Vec<i64> {
            let (dt, dims, raw) = &self.inits[name];
            assert_eq!(*dt, DT_INT64, "{name}");
            assert_eq!(dims, &[raw.len() as i64 / 8], "{name}: 1-D");
            raw.as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect()
        }
    }

    fn rewritten(bytes: &[u8], opts: &RewriteOptions) -> Rewrite {
        match rewrite(bytes, opts).unwrap() {
            Outcome::Rewritten(r) => r,
            other => panic!("expected a rewrite, got {other:?}"),
        }
    }

    #[test]
    fn reencode_is_byte_identical() {
        let hs = node(
            "HardSigmoid",
            "hs",
            &["x"],
            &["y"],
            &[attr_f("alpha", 0.25)],
        );
        let mut unknown = Writer::default();
        unknown.string(999, "future field");
        unknown.int(1000, -5);
        unknown.float(1001, 1.5);
        unknown.key(1002, 1);
        unknown.raw(&7u64.to_le_bytes());
        let g = G {
            nodes: vec![hs],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 3, -1, 4]))],
            outputs: vec![value_info("y", DT_FLOAT, None)],
            extra: unknown.buf.clone(),
            ..G::default()
        };
        let m = model(12, &g, &unknown.buf);
        assert_eq!(reencode(&m).unwrap(), m);
        // Nothing to rewrite: unchanged, no copy.
        let opts = RewriteOptions {
            hard_sigmoid: false,
            ..RewriteOptions::ALL
        };
        assert!(matches!(
            rewrite(&m, &opts).unwrap(),
            Outcome::Unchanged { ref skipped } if skipped.is_empty()
        ));
    }

    #[test]
    fn unknown_fields_and_untouched_nodes_are_preserved() {
        let mut unknown = Writer::default();
        unknown.string(999, "keep me");
        unknown.int(1000, 42);
        // An untouched node with an unknown field of its own.
        let mut relu = node("Relu", "relu", &["y"], &["z"], &[]);
        relu.extend_from_slice(&unknown.buf);
        let hs = node("HardSigmoid", "hs", &["x"], &["y"], &[]);
        let weights = tensor_f32("w", &[2], &[1.0, 2.0], DT_FLOAT);
        let g = G {
            nodes: vec![hs, relu.clone()],
            inits: vec![weights.clone()],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[1, 2]))],
            outputs: vec![value_info("z", DT_FLOAT, None)],
            extra: unknown.buf.clone(),
            ..G::default()
        };
        let m = model(13, &g, &unknown.buf);
        let r = rewritten(&m, &RewriteOptions::COREML);
        let contains = |hay: &[u8], needle: &[u8]| hay.windows(needle.len()).any(|w| w == needle);
        let out = parse_fields(&r.bytes).unwrap();
        // Model-level unknown fields, verbatim.
        assert!(
            out.iter()
                .any(|f| f.num == 999 && f.value == Value::Len(b"keep me"))
        );
        assert!(
            out.iter()
                .any(|f| f.num == 1000 && f.value == Value::Varint(42))
        );
        let gb = get_len(&out, MODEL_GRAPH).unwrap();
        let gf = parse_fields(gb).unwrap();
        assert!(
            gf.iter()
                .any(|f| f.num == 999 && f.value == Value::Len(b"keep me"))
        );
        assert!(gf.iter().any(|f| f.num == 1000));
        // The untouched node and the weights, byte for byte.
        assert!(contains(gb, &relu));
        assert!(contains(gb, &weights));
        assert_eq!(get_str(&gf, 2).unwrap(), "test");
        assert_eq!(get_int(&out, 1), Some(8));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip", "Relu"]);
        assert_eq!(
            v.metadata,
            [(
                "blue_onyx_prism.rewrite".to_string(),
                format!("v{REWRITER_VERSION}: HardSigmoid x1")
            )]
        );
    }

    #[test]
    fn hard_sigmoid_opset_11_plus_uses_clip_inputs() {
        let hs = node(
            "HardSigmoid",
            "/act/HardSigmoid",
            &["x"],
            &["y"],
            &[attr_f("alpha", 1.0 / 6.0), attr_f("beta", 0.5)],
        );
        let m = single(12, hs, &["y"], &[1, 4]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("HardSigmoid"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip"]);
        let (mul, add, clip) = (&v.nodes[0], &v.nodes[1], &v.nodes[2]);
        assert_eq!(mul.1[0], "x");
        assert_eq!(v.f32_of(&mul.1[1]), 1.0f32 / 6.0);
        assert_eq!(add.1[0], mul.2[0]);
        assert_eq!(v.f32_of(&add.1[1]), 0.5);
        assert_eq!(clip.1[0], add.2[0]);
        assert_eq!(v.f32_of(&clip.1[1]), 0.0);
        assert_eq!(v.f32_of(&clip.1[2]), 1.0);
        assert_eq!(clip.2, ["y"]);
        assert!(clip.3.is_empty());
        // Shared constants: 1/6, 0.5, 0 and 1.
        assert_eq!(v.inits.len(), 4);
        // Fresh names, never reusing graph names.
        assert!(mul.2[0].starts_with("y/bop_mul"));
    }

    #[test]
    fn hard_sigmoid_defaults_and_old_clip() {
        // alpha/beta default to 0.2/0.5; before opset 11 Clip takes min/max attributes.
        let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
        let m = single(10, hs, &["y"], &[2]);
        let v = view(&rewritten(&m, &RewriteOptions::COREML).bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip"]);
        assert_eq!(v.f32_of(&v.nodes[0].1[1]), 0.2);
        assert_eq!(v.f32_of(&v.nodes[1].1[1]), 0.5);
        let clip = &v.nodes[2];
        assert_eq!(clip.1.len(), 1);
        assert_eq!(
            clip.3,
            [
                ("min".to_string(), Some(0.0), vec![]),
                ("max".to_string(), Some(1.0), vec![])
            ]
        );
        assert_eq!(v.inits.len(), 2);
        // Before opset 7 (no broadcasting Mul/Add) nothing is rewritten.
        let hs = node("HardSigmoid", "", &["x"], &["y"], &[]);
        match rewrite(&single(6, hs, &["y"], &[2]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => {
                assert!(skipped[0].contains("opset 6"), "{skipped:?}")
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn hard_swish_multiplies_by_x() {
        let hs = node("HardSwish", "/act/HardSwish", &["x"], &["y"], &[]);
        let m = single(17, hs, &["y"], &[1, 8]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("HardSwish"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Mul", "Add", "Clip", "Mul"]);
        assert_eq!(v.f32_of(&v.nodes[0].1[1]), 1.0f32 / 6.0);
        assert_eq!(v.f32_of(&v.nodes[1].1[1]), 0.5);
        let last = &v.nodes[3];
        assert_eq!(last.1, ["x".to_string(), v.nodes[2].2[0].clone()]);
        assert_eq!(last.2, ["y"]);
        assert_ne!(v.nodes[2].2[0], "y");
    }

    #[test]
    fn float16_and_double_scalars_follow_the_input_type() {
        for (dt, size) in [(DT_FLOAT16, 2usize), (DT_DOUBLE, 8)] {
            let g = G {
                nodes: vec![node("HardSwish", "hs", &["x"], &["y"], &[])],
                inputs: vec![value_info("x", dt, Some(&[4]))],
                outputs: vec![value_info("y", dt, None)],
                ..G::default()
            };
            let v = view(&rewritten(&model(14, &g, &[]), &RewriteOptions::COREML).bytes);
            let (t, dims, raw) = &v.inits[&v.nodes[0].1[1]];
            assert_eq!((*t, dims.len(), raw.len()), (dt, 0, size));
            if dt == DT_FLOAT16 {
                assert_eq!(raw[..], 0x3155u16.to_le_bytes());
            } else {
                assert_eq!(
                    f64::from_le_bytes(raw[..].try_into().unwrap()),
                    f64::from(1.0f32 / 6.0)
                );
            }
        }
    }

    #[test]
    fn unknown_element_type_in_mixed_precision_model_is_skipped() {
        // The HardSigmoid input has no recorded type and the model also holds f16 data: the
        // scalar type cannot be chosen safely.
        let g = G {
            nodes: vec![
                node(
                    "Cast",
                    "c",
                    &["x"],
                    &["h"],
                    &[attr_i("to", i64::from(DT_FLOAT16))],
                ),
                node("HardSigmoid", "hs", &["h"], &["y"], &[]),
            ],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[4]))],
            outputs: vec![value_info("y", DT_FLOAT16, None)],
            ..G::default()
        };
        match rewrite(&model(13, &g, &[]), &RewriteOptions::COREML).unwrap() {
            Outcome::Unchanged { skipped } => {
                assert_eq!(skipped.len(), 1);
                assert!(
                    skipped[0].contains("element type of 'h' unknown"),
                    "{skipped:?}"
                );
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_attribute_form_becomes_slices() {
        // YOLOv5 head: Split(axis=4, split=[2,2,4]) on a 5-D tensor, opset 12.
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b", "c"],
            &[attr_i("axis", 4), attr_ints("split", &[2, 2, 4], false)],
        );
        let m = single(12, sp, &["a", "b", "c"], &[1, 3, 80, 80, 8]);
        let r = rewritten(&m, &RewriteOptions::COREML);
        assert_eq!(r.counts.get("Split"), Some(&1));
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Slice", "Slice", "Slice"]);
        let mut ranges = Vec::new();
        for (j, out) in ["a", "b", "c"].iter().enumerate() {
            let n = &v.nodes[j];
            assert_eq!(n.1.len(), 4);
            assert_eq!(n.1[0], "x");
            assert_eq!(n.2, [out.to_string()]);
            assert_eq!(v.i64s_of(&n.1[3]), [4]);
            ranges.push((v.i64s_of(&n.1[1])[0], v.i64s_of(&n.1[2])[0]));
        }
        assert_eq!(ranges, [(0, 2), (2, 4), (4, 8)]);
        // starts 0/2/4, ends 2/4/8 and axes 4, shared: 0,2,4,8 -> 4 tensors ([4] doubles as axes).
        assert_eq!(v.inits.len(), 4);
        // Packed `split` ints parse the same.
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", -1), attr_ints("split", &[3, 5], true)],
        );
        let v = view(&rewritten(&single(12, sp, &["a", "b"], &[8]), &RewriteOptions::COREML).bytes);
        assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [3]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [8]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[3]), [-1]);
    }

    #[test]
    fn split_before_opset_10_uses_slice_attributes() {
        let sp = node(
            "Split",
            "split",
            &["x"],
            &["a", "b"],
            &[attr_i("axis", 1), attr_ints("split", &[1, 3], false)],
        );
        let v = view(
            &rewritten(
                &single(9, sp, &["a", "b"], &[2, 4]),
                &RewriteOptions::COREML,
            )
            .bytes,
        );
        assert_eq!(v.ops(), ["Slice", "Slice"]);
        assert!(v.inits.is_empty());
        assert_eq!(v.nodes[1].1, ["x"]);
        assert_eq!(
            v.nodes[1].3,
            [
                ("starts".to_string(), None, vec![1]),
                ("ends".to_string(), None, vec![4]),
                ("axes".to_string(), None, vec![1])
            ]
        );
    }

    #[test]
    fn equal_split_needs_the_dimension() {
        // Opset 11 without `split`: equal parts of the known dimension.
        let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
        let v = view(
            &rewritten(
                &single(11, sp, &["a", "b"], &[1, 6]),
                &RewriteOptions::COREML,
            )
            .bytes,
        );
        assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [3]);
        assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [6]);
        // Unknown (symbolic) dimension: skipped with a reason.
        let sp = node("Split", "s", &["x"], &["a", "b"], &[attr_i("axis", 1)]);
        match rewrite(
            &single(11, sp, &["a", "b"], &[1, -1]),
            &RewriteOptions::COREML,
        )
        .unwrap()
        {
            Outcome::Unchanged { skipped } => {
                assert!(skipped[0].contains("needs its shape"), "{skipped:?}")
            }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_opset_13_input_from_initializer_or_constant() {
        for (raw, constant) in [(true, false), (false, false), (true, true)] {
            let sizes = tensor_i64("sizes", &[2, 6], raw);
            let mut g = G {
                nodes: vec![node(
                    "Split",
                    "s",
                    &["x", "sizes"],
                    &["a", "b"],
                    &[attr_i("axis", 0)],
                )],
                inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
                outputs: vec![
                    value_info("a", DT_FLOAT, None),
                    value_info("b", DT_FLOAT, None),
                ],
                ..G::default()
            };
            if constant {
                g.nodes.insert(
                    0,
                    node("Constant", "k", &[], &["sizes"], &[attr_t("value", sizes)]),
                );
            } else {
                g.inits.push(sizes);
            }
            let m = model(13, &g, &[]);
            // CoreML takes opset-13 Split as is.
            assert!(matches!(
                rewrite(&m, &RewriteOptions::COREML).unwrap(),
                Outcome::Unchanged { .. }
            ));
            let v = view(&rewritten(&m, &RewriteOptions::ALL).bytes);
            // The sizes tensor/Constant is dropped once no node reads it.
            assert_eq!(v.ops(), ["Slice", "Slice"], "raw {raw} constant {constant}");
            assert!(!v.inits.contains_key("sizes"));
            assert_eq!(v.i64s_of(&v.nodes[1].1[1]), [2]);
            assert_eq!(v.i64s_of(&v.nodes[1].1[2]), [8]);
        }
        // Still read by another node: kept.
        let g = G {
            nodes: vec![
                node("Split", "s", &["x", "sizes"], &["a", "b"], &[]),
                node("Identity", "i", &["sizes"], &["c"], &[]),
            ],
            inits: vec![tensor_i64("sizes", &[2, 6], true)],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            ..G::default()
        };
        let v = view(&rewritten(&model(13, &g, &[]), &RewriteOptions::ALL).bytes);
        assert!(v.inits.contains_key("sizes"));
        // Non-constant sizes: skipped.
        let g = G {
            nodes: vec![node("Split", "s", &["x", "dyn"], &["a", "b"], &[])],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[8]))],
            ..G::default()
        };
        match rewrite(&model(13, &g, &[]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => assert!(skipped[0].contains("not a constant")),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn split_opset_18_num_outputs() {
        // 10 into 3 parts: ceil chunks 4, 4, 2.
        let sp = node(
            "Split",
            "s",
            &["x"],
            &["a", "b", "c"],
            &[attr_i("axis", -1), attr_i("num_outputs", 3)],
        );
        let v = view(
            &rewritten(
                &single(18, sp, &["a", "b", "c"], &[2, 10]),
                &RewriteOptions::ALL,
            )
            .bytes,
        );
        let r: Vec<_> = v
            .nodes
            .iter()
            .map(|n| (v.i64s_of(&n.1[1])[0], v.i64s_of(&n.1[2])[0]))
            .collect();
        assert_eq!(r, [(0, 4), (4, 8), (8, 10)]);
        // From value_info rather than graph inputs.
        let g = G {
            nodes: vec![
                node("Relu", "r", &["x"], &["t"], &[]),
                node(
                    "Split",
                    "s",
                    &["t"],
                    &["a", "b"],
                    &[attr_i("axis", 0), attr_i("num_outputs", 2)],
                ),
            ],
            inputs: vec![value_info("x", DT_FLOAT, None)],
            value_info: vec![value_info("t", DT_FLOAT, Some(&[6, 2]))],
            ..G::default()
        };
        let v = view(&rewritten(&model(18, &g, &[]), &RewriteOptions::ALL).bytes);
        assert_eq!(v.ops(), ["Relu", "Slice", "Slice"]);
        assert_eq!(v.i64s_of(&v.nodes[2].1[1]), [3]);
        // No shape anywhere: skipped, logged.
        let sp = node(
            "Split",
            "s",
            &["x"],
            &["a", "b"],
            &[attr_i("num_outputs", 2)],
        );
        let g = G {
            nodes: vec![sp],
            inputs: vec![value_info("x", DT_FLOAT, None)],
            ..G::default()
        };
        match rewrite(&model(18, &g, &[]), &RewriteOptions::ALL).unwrap() {
            Outcome::Unchanged { skipped } => assert!(skipped[0].starts_with("Split 's'")),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn flatten_becomes_reshape() {
        let case = |axis: Option<i64>, dims: &[i64]| {
            let attrs: Vec<_> = axis.map(|a| attr_i("axis", a)).into_iter().collect();
            let f = node("Flatten", "f", &["x"], &["y"], &attrs);
            rewrite(&single(13, f, &["y"], dims), &RewriteOptions::ALL).unwrap()
        };
        let shape_of = |o: Outcome| match o {
            Outcome::Rewritten(r) => {
                let v = view(&r.bytes);
                assert_eq!(v.ops(), ["Reshape"]);
                assert_eq!(v.nodes[0].1[0], "x");
                assert_eq!(v.nodes[0].2, ["y"]);
                v.i64s_of(&v.nodes[0].1[1])
            }
            o => panic!("{o:?}"),
        };
        assert_eq!(shape_of(case(None, &[-1, 3, 4])), [0, -1]);
        let f = node("Flatten", "f", &["x"], &["y"], &[]);
        assert!(matches!(
            rewrite(&single(13, f, &["y"], &[1, 4]), &RewriteOptions::COREML).unwrap(),
            Outcome::Unchanged { .. }
        ));
        assert_eq!(shape_of(case(Some(0), &[-1, 3])), [1, -1]);
        assert_eq!(shape_of(case(Some(2), &[2, 3, 4, 5])), [6, -1]);
        assert_eq!(shape_of(case(Some(-1), &[2, 3, 4])), [6, -1]);
        assert_eq!(shape_of(case(Some(-2), &[-1, 3, 4])), [0, -1]);
        match case(Some(2), &[-1, 3, 4]) {
            Outcome::Unchanged { skipped } => assert!(skipped[0].contains("static leading")),
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn external_data_is_left_alone() {
        let ext = msg(|w| {
            w.int(TENSOR_DIMS, 2);
            w.int(TENSOR_DATA_TYPE, i64::from(DT_FLOAT));
            w.string(TENSOR_NAME, "w");
            let e = msg(|e| {
                e.string(ENTRY_KEY, "location");
                e.string(ENTRY_VALUE, "weights.bin");
            });
            w.bytes(TENSOR_EXTERNAL_DATA, &e);
            w.int(TENSOR_DATA_LOCATION, 1);
        });
        let g = G {
            nodes: vec![node("HardSigmoid", "hs", &["x"], &["y"], &[])],
            inits: vec![ext],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[2]))],
            ..G::default()
        };
        assert!(matches!(
            rewrite(&model(13, &g, &[]), &RewriteOptions::ALL).unwrap(),
            Outcome::ExternalData
        ));
    }

    #[test]
    fn names_never_collide_and_other_domains_are_ignored() {
        let g = G {
            nodes: vec![
                node(
                    "Relu",
                    "bop_rewrite/const_0",
                    &["x"],
                    &["bop_rewrite/const_0"],
                    &[],
                ),
                node("HardSigmoid", "hs", &["bop_rewrite/const_0"], &["y"], &[]),
                msg(|w| {
                    w.string(NODE_INPUT, "y");
                    w.string(NODE_OUTPUT, "z");
                    w.string(NODE_OP_TYPE, "HardSigmoid");
                    w.string(NODE_DOMAIN, "com.example");
                }),
            ],
            inputs: vec![value_info("x", DT_FLOAT, Some(&[2]))],
            ..G::default()
        };
        let r = rewritten(&model(13, &g, &[]), &RewriteOptions::ALL);
        assert_eq!(
            r.counts.get("HardSigmoid"),
            Some(&1),
            "custom domain untouched"
        );
        let v = view(&r.bytes);
        assert_eq!(v.ops(), ["Relu", "Mul", "Add", "Clip", "HardSigmoid"]);
        assert!(!v.inits.contains_key("bop_rewrite/const_0"));
        assert!(v.inits.contains_key("bop_rewrite/const_0_1"));
    }

    #[test]
    fn rewrite_cached_writes_once_and_keeps_the_source() {
        let dir =
            std::env::temp_dir().join(format!("bop-onnx-rewrite-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("tiny.onnx");
        let m = single(
            12,
            node("HardSigmoid", "hs", &["x"], &["y"], &[]),
            &["y"],
            &[2],
        );
        std::fs::write(&src, &m).unwrap();
        let cache = dir.join("cache").join("coreml");
        let first = rewrite_cached(&src, &cache, &RewriteOptions::COREML).unwrap();
        let Cached::Rewritten {
            path,
            created,
            counts,
            ..
        } = first
        else {
            panic!("{first:?}")
        };
        assert!(created);
        assert_eq!(counts.get("HardSigmoid"), Some(&1));
        assert_eq!(path, cache.join(cache_file_name(&src, &sha256_hex(&m))));
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("tiny-")
        );
        assert_eq!(std::fs::read(&src).unwrap(), m, "source untouched");
        let written = std::fs::read(&path).unwrap();
        assert_eq!(view(&written).ops(), ["Mul", "Add", "Clip"]);
        let again = rewrite_cached(&src, &cache, &RewriteOptions::COREML).unwrap();
        assert!(
            matches!(again, Cached::Rewritten { created: false, path: ref p, .. } if *p == path)
        );
        // No temporary files left behind.
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        // A model with nothing to do is not copied.
        let plain = dir.join("plain.onnx");
        std::fs::write(
            &plain,
            single(12, node("Relu", "r", &["x"], &["y"], &[]), &["y"], &[2]),
        )
        .unwrap();
        assert!(matches!(
            rewrite_cached(&plain, &cache, &RewriteOptions::COREML).unwrap(),
            Cached::Original { .. }
        ));
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn garbage_is_an_error() {
        assert!(rewrite(b"not a model at all \xff\xff", &RewriteOptions::COREML).is_err());
        assert!(rewrite(&[], &RewriteOptions::COREML).is_err(), "no graph");
    }

    #[test]
    fn cache_names() {
        let sha = "0123456789abcdef0123";
        assert_eq!(
            cache_file_name(Path::new("/m/IPcam-general.onnx"), sha),
            format!("IPcam-general-0123456789abcdef-v{REWRITER_VERSION}.onnx")
        );
    }
}
