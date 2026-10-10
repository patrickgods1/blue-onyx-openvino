//! In-memory log ring for the web UI's Logs page (`GET /logs`, `GET /logs.json`).
//!
//! [`LogBuffer`] is a `tracing` [`Layer`] that keeps the last [`DEFAULT_CAPACITY`] events
//! (timestamp, level, target, message plus fields) in a bounded ring. Every event gets a
//! monotonically increasing sequence number, so a client polls with `after=<last seq>` and only
//! receives what is new. `cli::init_logging_with` always installs it under the reloadable level
//! filter, so it holds exactly what the configured level emits.

use serde::Serialize;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// Events kept in memory.
pub const DEFAULT_CAPACITY: usize = 2000;
/// Most entries one `/logs.json` reply returns.
pub const MAX_PAGE: usize = 2000;

/// One captured event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogEntry {
    /// 1-based, increasing by one per event (also across evictions).
    pub seq: u64,
    /// RFC 3339 UTC with milliseconds.
    pub ts: String,
    /// `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`.
    pub level: &'static str,
    pub target: String,
    /// The message followed by the other fields as `key=value`.
    pub message: String,
}

#[derive(Debug)]
struct Ring {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    /// Sequence number of the last pushed entry (0 = none yet).
    last_seq: u64,
}

/// Bounded log ring; cheap to clone (shared).
#[derive(Debug, Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<Ring>>,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// A page of entries for `/logs.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogPage {
    pub entries: Vec<LogEntry>,
    /// The newest sequence number in the buffer: poll again with `after=last`.
    pub last: u64,
    /// Oldest sequence number still held (entries before it were evicted).
    pub first: u64,
    /// Entries between `after` and `first` were evicted before the client saw them.
    pub truncated: bool,
}

/// `error` < `warn` < `info` < `debug` < `trace` in verbosity; parse a level name (any case).
pub fn parse_level(s: &str) -> Option<Level> {
    match s.trim().to_ascii_lowercase().as_str() {
        "error" => Some(Level::ERROR),
        "warn" | "warning" => Some(Level::WARN),
        "info" => Some(Level::INFO),
        "debug" => Some(Level::DEBUG),
        "trace" => Some(Level::TRACE),
        _ => None,
    }
}

fn level_from_str(s: &str) -> Level {
    parse_level(s).unwrap_or(Level::TRACE)
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(Mutex::new(Ring {
                entries: VecDeque::with_capacity(capacity.min(4096)),
                capacity,
                last_seq: 0,
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Ring> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Append an entry (evicting the oldest when full); returns its sequence number.
    pub fn push(&self, level: Level, target: &str, message: String) -> u64 {
        let ts = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        let mut r = self.lock();
        r.last_seq += 1;
        let seq = r.last_seq;
        if r.entries.len() >= r.capacity {
            r.entries.pop_front();
        }
        r.entries.push_back(LogEntry {
            seq,
            ts,
            level: level.as_str(),
            target: target.to_string(),
            message,
        });
        seq
    }

    /// Number of entries held.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entries with `seq > after` at `min` severity or more severe (None = all), oldest first,
    /// at most `limit` (the newest ones when there are more).
    pub fn since(&self, after: u64, min: Option<Level>, limit: usize) -> LogPage {
        let r = self.lock();
        let first = r.entries.front().map_or(r.last_seq + 1, |e| e.seq);
        let mut entries: Vec<LogEntry> = r
            .entries
            .iter()
            .filter(|e| e.seq > after)
            // tracing orders levels by verbosity: ERROR < WARN < ... < TRACE.
            .filter(|e| min.is_none_or(|m| level_from_str(e.level) <= m))
            .cloned()
            .collect();
        if entries.len() > limit {
            entries.drain(..entries.len() - limit);
        }
        LogPage {
            entries,
            last: r.last_seq,
            first,
            truncated: after + 1 < first && after < r.last_seq,
        }
    }
}

/// Collects the `message` field and the other fields of an event.
#[derive(Default)]
struct FieldText {
    message: String,
    fields: String,
}

impl Visit for FieldText {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.fields, " {}={}", field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else if !field.name().starts_with("log.") {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

impl<S: Subscriber> Layer<S> for LogBuffer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut v = FieldText::default();
        event.record(&mut v);
        let mut text = v.message;
        if !v.fields.is_empty() {
            if text.is_empty() {
                text.push_str(v.fields.trim_start());
            } else {
                text.push_str(&v.fields);
            }
        }
        let meta = event.metadata();
        self.push(*meta.level(), meta.target(), text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn ring_keeps_the_newest_entries() {
        let b = LogBuffer::new(3);
        for i in 1..=5 {
            assert_eq!(b.push(Level::INFO, "t", format!("m{i}")), i);
        }
        assert_eq!(b.len(), 3);
        let p = b.since(0, None, 100);
        let msgs: Vec<_> = p.entries.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(msgs, ["m3", "m4", "m5"]);
        assert_eq!((p.first, p.last), (3, 5));
        assert!(p.truncated, "entries 1..2 were evicted unseen");
    }

    #[test]
    fn cursor_returns_only_new_entries() {
        let b = LogBuffer::new(10);
        b.push(Level::INFO, "a", "one".into());
        b.push(Level::WARN, "a", "two".into());
        let p = b.since(0, None, 100);
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.last, 2);
        assert!(!p.truncated);
        let p = b.since(p.last, None, 100);
        assert!(p.entries.is_empty());
        assert_eq!(p.last, 2);
        b.push(Level::ERROR, "a", "three".into());
        let p = b.since(2, None, 100);
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.entries[0].seq, 3);
        assert_eq!(p.entries[0].level, "ERROR");
        // A cursor ahead of the buffer (server restarted) returns nothing.
        assert!(b.since(99, None, 100).entries.is_empty());
    }

    #[test]
    fn level_filter_and_limit() {
        let b = LogBuffer::new(10);
        for (l, m) in [
            (Level::TRACE, "t"),
            (Level::DEBUG, "d"),
            (Level::INFO, "i"),
            (Level::WARN, "w"),
            (Level::ERROR, "e"),
        ] {
            b.push(l, "x", m.into());
        }
        let msgs =
            |p: LogPage| -> Vec<String> { p.entries.into_iter().map(|e| e.message).collect() };
        assert_eq!(msgs(b.since(0, Some(Level::WARN), 100)), ["w", "e"]);
        assert_eq!(msgs(b.since(0, Some(Level::INFO), 100)), ["i", "w", "e"]);
        assert_eq!(msgs(b.since(0, None, 2)), ["w", "e"]);
        assert_eq!(parse_level("Warning"), Some(Level::WARN));
        assert_eq!(parse_level("loud"), None);
    }

    #[test]
    fn layer_captures_message_and_fields() {
        let b = LogBuffer::new(10);
        let sub = tracing_subscriber::registry().with(b.clone());
        tracing::subscriber::with_default(sub, || {
            tracing::info!(model = "m1", count = 3, "loaded");
            tracing::warn!(target: "custom", "plain");
        });
        let p = b.since(0, None, 10);
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.entries[0].message, "loaded model=m1 count=3");
        assert_eq!(p.entries[0].level, "INFO");
        assert_eq!(p.entries[1].target, "custom");
        assert_eq!(p.entries[1].message, "plain");
        assert!(p.entries[0].ts.ends_with('Z'));
    }
}
