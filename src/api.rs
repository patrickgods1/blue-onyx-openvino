//! Wire structs for the CodeProject.AI / DeepStack compatible API that Blue Iris speaks.
//! Shapes copied from blue-onyx (MIT). Field names are part of the contract: keep them.

use anyhow::{Context, anyhow};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Debug;

/// A detection request as received over HTTP (multipart `image` + optional `min_confidence`).
#[derive(Default, Clone)]
pub struct VisionDetectionRequest {
    /// 0.0 disables the override and the model's configured threshold applies.
    pub min_confidence: f32,
    pub image_data: Bytes,
    pub image_name: String,
}

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct VisionDetectionResponse {
    pub success: bool,
    pub message: String,
    pub error: Option<String>,
    pub predictions: Vec<Prediction>,
    pub count: i32,
    /// "detect" for the default model, "custom" for a named model.
    pub command: String,
    pub moduleId: String,
    /// e.g. "OpenVINO GPU (Intel(R) UHD Graphics 630)"
    pub executionProvider: String,
    pub canUseGPU: bool,
    pub inferenceMs: i32,
    pub processMs: i32,
    pub analysisRoundTripMs: i32,
}

impl VisionDetectionResponse {
    pub fn error(msg: impl Into<String>) -> Self {
        let msg = msg.into();
        Self {
            success: false,
            message: msg.clone(),
            error: Some(msg),
            moduleId: crate::MODULE_ID.to_string(),
            ..Default::default()
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Prediction {
    pub x_max: usize,
    pub x_min: usize,
    pub y_max: usize,
    pub y_min: usize,
    pub confidence: f32,
    pub label: String,
}

impl Debug for Prediction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prediction")
            .field("label", &self.label)
            .field("confidence", &self.confidence)
            .field("box", &(self.x_min, self.y_min, self.x_max, self.y_max))
            .finish()
    }
}

#[allow(non_snake_case)]
#[derive(Serialize, Default, Debug)]
#[serde(rename_all = "camelCase")]
pub struct VisionCustomListResponse {
    pub success: bool,
    pub models: Vec<String>,
    pub moduleId: String,
    pub moduleName: String,
    pub command: String,
    pub statusData: Option<String>,
    pub inferenceDevice: String,
    pub analysisRoundTripMs: i32,
    pub processedBy: String,
    pub timestampUTC: String,
}

#[allow(non_snake_case)]
#[derive(Serialize, Default, Debug)]
#[serde(rename_all = "camelCase")]
pub struct StatusUpdateResponse {
    pub success: bool,
    pub message: String,
    pub version: Option<VersionInfo>,
    pub current: VersionInfo,
    pub latest: VersionInfo,
    pub updateAvailable: bool,
}

#[allow(non_snake_case)]
#[derive(Serialize, Default, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VersionInfo {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
    pub preRelease: Option<String>,
    pub securityUpdate: bool,
    pub build: u32,
    pub file: String,
    pub releaseNotes: String,
}

impl VersionInfo {
    pub fn parse(version_str: &str, release_notes: Option<String>) -> anyhow::Result<Self> {
        let v = version_str.trim().trim_start_matches('v');
        let parts: Vec<_> = v.split('.').collect();
        let seg = |i: usize, name: &str| -> anyhow::Result<u8> {
            parts
                .get(i)
                .ok_or_else(|| anyhow!("Missing {name} version segment"))?
                .parse()
                .with_context(|| format!("Failed to parse {name} version from '{version_str}'"))
        };
        Ok(Self {
            major: seg(0, "major")?,
            minor: seg(1, "minor")?,
            patch: seg(2, "patch")?,
            releaseNotes: release_notes.unwrap_or_default(),
            ..Default::default()
        })
    }
}

impl PartialEq for VersionInfo {
    fn eq(&self, other: &Self) -> bool {
        self.major == other.major && self.minor == other.minor && self.patch == other.patch
    }
}
impl Eq for VersionInfo {}
impl PartialOrd for VersionInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for VersionInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

#[cfg(test)]
mod tests {
    use super::VersionInfo;

    #[test]
    fn version_parse_and_order() {
        let a = VersionInfo::parse("1.2.3", None).unwrap();
        let b = VersionInfo::parse("v1.2.4", None).unwrap();
        let c = VersionInfo::parse("2.0.0", None).unwrap();
        assert!(a < b && b < c);
        assert_eq!(
            a,
            VersionInfo::parse("1.2.3", Some("notes".into())).unwrap()
        );
        assert!(VersionInfo::parse("1.2", None).is_err());
    }
}
