"""Latency benchmark: {baseline, optimised} x {FP32, INT8} x {1 thread, all cores}, plus GC off.

Every run is a fresh `python -m kws.stream` process on the 1-hour test stream, so peak RSS is
per run and nothing is shared between repetitions. Each configuration is repeated `--reps` times
(default 5); the results file keeps every repetition and the median across repetitions of each
percentile. A separate short run per (mode, model) traces Python allocations with tracemalloc
(1,000 hops after warm-up); those runs are never used for timings.

    python -m tools.bench                 # full matrix, about 1 hour on the reference laptop
    python -m tools.bench --reps 1 --stream data/streams/val.wav     # quick smoke run

Writes results/latency_<mode>_<model>_<threads>[_nogc].json. Timings are Python on the host CPU.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from kws.envinfo import REPO_ROOT, write_results
from tools.make_streams import stream_paths

RESULTS_DIR = REPO_ROOT / "results"
MODELS = {"fp32": REPO_ROOT / "models" / "dscnn_fp32.onnx",
          "int8": REPO_ROOT / "models" / "dscnn_int8.onnx"}  # fmt: skip
THREAD_LABELS = {1: "1t", 0: "allt"}
ALLOC_HOPS = 1000
PERCENTILES = ("p50", "p95", "p99", "max")
STAGE_KEYS = ("features_ms", "inference_ms", "post_ms", "total_per_hop_ms")


def run_once(
    mode: str,
    model: str,
    threads: int,
    gc_on: bool,
    wav: Path,
    threshold: float,
    smooth: int,
    out_dir: Path,
    alloc_hops: int = 0,
    pin_cpu: int | None = None,
) -> dict[str, Any]:
    out = out_dir / "timings.json"
    cmd = [
        sys.executable, "-m", "kws.stream",
        "--model", str(MODELS[model]), "--wav", str(wav),
        "--threshold", str(threshold), "--smooth", str(smooth), "--infer-every", "5",
        "--threads", str(threads), "--mode", mode, "--timings", str(out),
    ]  # fmt: skip
    if not gc_on:
        cmd.append("--no-gc")
    if pin_cpu is not None:
        cmd += ["--pin-cpu", str(pin_cpu)]
    if alloc_hops:
        cmd += ["--alloc-hops", str(alloc_hops)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True)
    data: dict[str, Any] = json.loads(out.read_text())
    out.unlink()
    return data


def aggregate(reps: list[dict[str, Any]]) -> dict[str, Any]:
    """Median across repetitions of each percentile, with min and max across repetitions."""
    agg: dict[str, Any] = {}
    for stage in STAGE_KEYS:
        agg[stage] = {}
        for p in PERCENTILES:
            vals = [r["timings"][stage][p] for r in reps]
            agg[stage][p] = {
                "median": statistics.median(vals), "min": min(vals), "max": max(vals),
            }  # fmt: skip
    misses = [r["timings"]["deadline_misses"] for r in reps]
    rss = [r["peak_rss_bytes"] for r in reps]
    agg["deadline_misses"] = {"median": statistics.median(misses), "min": min(misses),
                              "max": max(misses)}  # fmt: skip
    agg["peak_rss_bytes"] = {"median": statistics.median(rss), "min": min(rss), "max": max(rss)}
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--stream", type=Path, default=None, help="default: the 1-hour test stream")
    p.add_argument("--val-results", type=Path, default=RESULTS_DIR / "stream_eval_val.json")
    p.add_argument(
        "--pin-cpu", type=int, default=2,
        help="logical CPU for the 1-thread runs (a performance core on hybrid CPUs); -1 = unpinned",
    )  # fmt: skip
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    chosen = json.loads(args.val_results.read_text())["chosen"]
    threshold, smooth = chosen["threshold"], chosen["smooth"]
    pin = None if args.pin_cpu < 0 else args.pin_cpu
    wav = args.stream or stream_paths("test")[0]

    # (mode, model, threads, gc_on); GC off only for the optimised, 1-thread configurations.
    configs = [
        (mode, model, threads, True)
        for mode in ("baseline", "optimised")
        for model in ("fp32", "int8")
        for threads in (1, 0)
    ] + [("optimised", model, 1, False) for model in ("fp32", "int8")]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for mode, model, threads, gc_on in configs:
            name = f"{mode}_{model}_{THREAD_LABELS[threads]}" + ("" if gc_on else "_nogc")
            reps = []
            for i in range(args.reps):
                reps.append(
                    run_once(
                        mode,
                        model,
                        threads,
                        gc_on,
                        wav,
                        threshold,
                        smooth,
                        tmp_dir,
                        pin_cpu=pin if threads == 1 else None,
                    )
                )
                t = reps[-1]["timings"]
                print(
                    f"{name} rep {i + 1}/{args.reps}: inference p50 "
                    f"{t['inference_ms']['p50']:.3f} ms, "
                    f"hop p99 {t['total_per_hop_ms']['p99']:.3f} ms"
                )
            alloc = None
            if gc_on and threads == 1:
                traced = run_once(
                    mode,
                    model,
                    1,
                    True,
                    wav,
                    threshold,
                    smooth,
                    tmp_dir,
                    alloc_hops=ALLOC_HOPS,
                    pin_cpu=pin,
                )
                alloc = traced["allocations"]
            write_results(
                args.out_dir / f"latency_{name}.json",
                {
                    "mode": mode,
                    "model": MODELS[model].name,
                    "model_bytes": MODELS[model].stat().st_size,
                    "stream": wav.name,
                    "stream_seconds": reps[0]["stream_seconds"],
                    "settings": {
                        **reps[0]["settings"],
                        "threads": threads,
                        "repetitions": args.reps,
                    },
                    "aggregate": aggregate(reps),
                    "allocations": alloc,
                    "allocations_note": (
                        f"Python (tracemalloc) high-water mark above the start-of-hop level, over "
                        f"{ALLOC_HOPS} hops after warm-up; ORT native allocations are not visible"
                    ),
                    "repetitions": [
                        {"timings": r["timings"], "peak_rss_bytes": r["peak_rss_bytes"]}
                        for r in reps
                    ],
                    "note": "Python-on-host-CPU timings; not device or accelerator latency. "
                    "threads 0 = ORT intra-op default (all cores)",
                },
                ort_threads={"intra_op_num_threads": threads, "inter_op_num_threads": 1},
            )


if __name__ == "__main__":
    main()
