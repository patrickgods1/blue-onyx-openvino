//! Class-name loading from YAML label files.

use std::path::Path;

use anyhow::{Context, Result};
use serde_yaml_ng::Value;

/// Load class names from a YAML file. Accepted forms:
/// - `NAMES:` / `names:` as a list (`- person`),
/// - `names:` as an index map (`{0: person, 1: bicycle}`, Ultralytics `metadata.yaml`), sorted by key,
/// - a bare top-level list.
pub fn load_class_names(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading class names from {}", path.display()))?;
    parse_class_names(&text).with_context(|| format!("parsing class names in {}", path.display()))
}

/// Parse class names from YAML text (see [`load_class_names`]).
pub fn parse_class_names(text: &str) -> Result<Vec<String>> {
    let root: Value = serde_yaml_ng::from_str(text).context("invalid YAML")?;
    let names = match &root {
        Value::Sequence(_) => &root,
        Value::Mapping(m) => m
            .get("NAMES")
            .or_else(|| m.get("names"))
            .or_else(|| m.get("Names"))
            .context("no `NAMES:` or `names:` key")?,
        _ => anyhow::bail!("expected a mapping with a `names` key or a list"),
    };
    match names {
        Value::Sequence(seq) => seq.iter().map(scalar_to_string).collect(),
        Value::Mapping(map) => {
            let mut pairs = map
                .iter()
                .map(|(k, v)| {
                    let idx = match k {
                        Value::Number(n) => n.as_u64(),
                        Value::String(s) => s.trim().parse::<u64>().ok(),
                        _ => None,
                    }
                    .with_context(|| {
                        format!("class map key {k:?} is not a non-negative integer")
                    })?;
                    Ok((idx, scalar_to_string(v)?))
                })
                .collect::<Result<Vec<_>>>()?;
            pairs.sort_by_key(|(i, _)| *i);
            Ok(pairs.into_iter().map(|(_, n)| n).collect())
        }
        other => anyhow::bail!("`names` must be a list or an index map, got {other:?}"),
    }
}

fn scalar_to_string(v: &Value) -> Result<String> {
    Ok(match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        other => anyhow::bail!("class name {other:?} is not a scalar"),
    })
}

/// The 80 COCO class names bundled with the binary.
pub fn coco80() -> Vec<String> {
    parse_class_names(include_str!("../../assets/coco_classes.yaml"))
        .expect("bundled coco_classes.yaml is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_form() {
        let n = parse_class_names("NAMES:\n  - person\n  - car\n  - dog\n").unwrap();
        assert_eq!(n, ["person", "car", "dog"]);
        let n = parse_class_names("names: [a, b]\n").unwrap();
        assert_eq!(n, ["a", "b"]);
    }

    #[test]
    fn map_form() {
        let yaml =
            "description: x\nnames:\n  2: dog\n  0: person\n  1: bicycle\n  10: fire hydrant\n";
        let n = parse_class_names(yaml).unwrap();
        assert_eq!(n, ["person", "bicycle", "dog", "fire hydrant"]);
    }

    #[test]
    fn from_file_and_errors() {
        let dir = std::env::temp_dir().join(format!("bo-classes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("c.yaml");
        std::fs::write(&p, "names:\n  0: a\n  1: b\n").unwrap();
        assert_eq!(load_class_names(&p).unwrap(), ["a", "b"]);
        assert!(load_class_names(&dir.join("missing.yaml")).is_err());
        std::fs::remove_dir_all(&dir).ok();
        assert!(parse_class_names("other: 1\n").is_err());
        assert!(parse_class_names("names: 5\n").is_err());
    }

    #[test]
    fn coco() {
        let n = coco80();
        assert_eq!(n.len(), 80);
        assert_eq!(n[0], "person");
        assert_eq!(n[2], "car");
    }
}
