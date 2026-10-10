//! Readiness state of a loaded model, shared between the loader/worker threads and HTTP handlers.

use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// Model is being read/compiled, waits for a download, or is lazy and not yet requested.
    Initializing,
    Ready,
    Failed(String),
}

#[derive(Debug)]
struct Inner {
    state: ModelState,
    /// What an `Initializing` model is doing, e.g. "downloading OpenVINO runtime 42% (18/44 MB)".
    detail: Option<String>,
}

#[derive(Clone, Debug)]
pub struct StateHandle(Arc<RwLock<Inner>>);

impl Default for StateHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHandle {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(Inner {
            state: ModelState::Initializing,
            detail: None,
        })))
    }

    pub fn get(&self) -> ModelState {
        self.0
            .read()
            .map(|s| s.state.clone())
            .unwrap_or(ModelState::Failed("poisoned".into()))
    }

    /// Set the state; clears the detail.
    pub fn set(&self, s: ModelState) {
        if let Ok(mut g) = self.0.write() {
            g.state = s;
            g.detail = None;
        }
    }

    /// `Initializing` with a detail message (download progress, waiting for a restart, ...).
    pub fn set_initializing(&self, detail: impl Into<String>) {
        if let Ok(mut g) = self.0.write() {
            g.state = ModelState::Initializing;
            g.detail = Some(detail.into());
        }
    }

    /// The detail of an `Initializing` model, if any.
    pub fn detail(&self) -> Option<String> {
        self.0
            .read()
            .ok()
            .filter(|g| g.state == ModelState::Initializing)
            .and_then(|g| g.detail.clone())
    }

    pub fn is_ready(&self) -> bool {
        self.get() == ModelState::Ready
    }

    /// "Initializing", "Initializing (downloading ...)", "Ready" or "Failed: <reason>".
    pub fn describe(&self) -> String {
        match (self.get(), self.detail()) {
            (ModelState::Initializing, Some(d)) => format!("Initializing ({d})"),
            (ModelState::Initializing, None) => "Initializing".into(),
            (ModelState::Ready, _) => "Ready".into(),
            (ModelState::Failed(m), _) => format!("Failed: {m}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_only_while_initializing() {
        let s = StateHandle::new();
        assert_eq!(s.describe(), "Initializing");
        s.set_initializing("downloading OpenVINO runtime 42% (18/44 MB)");
        assert_eq!(
            s.describe(),
            "Initializing (downloading OpenVINO runtime 42% (18/44 MB))"
        );
        s.set(ModelState::Ready);
        assert_eq!(s.detail(), None);
        assert_eq!(s.describe(), "Ready");
    }
}
