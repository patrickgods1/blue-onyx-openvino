//! Benchmark: compile each model, then time the production per-request pipeline
//! (decode -> preprocess -> infer -> postprocess, as in `worker.rs`) over N repeats.
//! Thin CLI over [`blue_onyx_prism::benchmark`]; also available as `blue-onyx-prism benchmark`.
//!
//! ```text
//! blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --device CPU --repeat 20
//! blue-onyx-prism-benchmark --model models/yolo26s.xml --compare-cpu      # GPU vs CPU + confidence diff
//! blue-onyx-prism-benchmark --json                                         # every enabled model in the config
//! blue-onyx-prism-benchmark --all-devices --apply                          # pick the best device per model
//! ```

use blue_onyx_prism::benchmark::cli::{BenchArgs, main as bench_main};
use clap::Parser;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "blue-onyx-prism-benchmark",
    version = env!("CARGO_PKG_VERSION"),
    about = "Benchmark Blue Onyx Prism models with the production pre/post-processing pipeline"
)]
struct Cli {
    #[command(flatten)]
    args: BenchArgs,
}

fn main() -> ExitCode {
    bench_main(Cli::parse().args, None)
}
