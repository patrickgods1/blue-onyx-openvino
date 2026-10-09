//! Readiness state of a loaded model, shared between the loader/worker threads and HTTP handlers.

use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelState {
    /// Model is being read/compiled (or is lazy and not yet requested).
    Initializing,
    Ready,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct StateHandle(Arc<RwLock<ModelState>>);

impl Default for StateHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHandle {
    pub fn new() -> Self {
        Self(Arc::new(RwLock::new(ModelState::Initializing)))
    }
    pub fn get(&self) -> ModelState {
        self.0
            .read()
            .map(|s| s.clone())
            .unwrap_or(ModelState::Failed("poisoned".into()))
    }
    pub fn set(&self, s: ModelState) {
        if let Ok(mut g) = self.0.write() {
            *g = s;
        }
    }
    pub fn is_ready(&self) -> bool {
        self.get() == ModelState::Ready
    }
}
