//! Release update check against the project's GitHub releases.

use crate::api::{StatusUpdateResponse, VersionInfo};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::time::Duration;

const LATEST_URL: &str =
    "https://api.github.com/repos/patrickgods1/blue-onyx-openvino/releases/latest";

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    body: Option<String>,
}

async fn fetch_latest() -> Result<Release> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;
    let resp = client
        .get(LATEST_URL)
        .header(
            reqwest::header::USER_AGENT,
            format!("blue-onyx-openvino/{}", crate::VERSION),
        )
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .context("requesting latest release")?
        .error_for_status()
        .context("release server returned an error")?;
    resp.json::<Release>().await.context("parsing release JSON")
}

/// Compare `current` against the latest release. Never fails: errors are reported through
/// `success: false` and `message`.
pub async fn check_update(current: &str) -> StatusUpdateResponse {
    let run = async {
        let current_v = VersionInfo::parse(current, None)?;
        let rel = fetch_latest().await?;
        let latest = VersionInfo::parse(&rel.tag_name, rel.body)?;
        Ok::<_, anyhow::Error>((current_v, latest))
    };
    match run.await {
        Ok((current_v, latest)) => StatusUpdateResponse {
            success: true,
            message: "Update check completed".to_string(),
            updateAvailable: latest > current_v,
            version: Some(latest.clone()),
            current: current_v,
            latest,
        },
        Err(e) => StatusUpdateResponse {
            success: false,
            message: format!("{e:#}"),
            ..Default::default()
        },
    }
}
