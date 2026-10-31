"""Streaming keyword spotting loop (optimised mode).

A 16 kHz mono int16 WAV is fed in hops of 320 samples, as if it arrived live. Per hop:

  1. StreamingMFCC turns the newest 320 samples (plus the 320 held over) into one frame.
  2. The frame is normalised into the FeatureRingBuffer.
  3. Every `infer_every` hops, once 49 frames exist, the buffer is copied in time order into the
     ORT input array, the model runs through IOBinding, and the Detector sees the logits.

Everything is preallocated before the loop. Detection time is the stream time (samples consumed
so far) of the inference that fired; the 49 frames cover exactly the last 16,000 samples.

Baseline mode does the same job the naive way: at every inference it recomputes all 49 frames
from the last 16,000 samples with the batch reference, builds fresh arrays and calls plain
`session.run`. The detector and the inference schedule are identical, so the two modes differ
only in how features and inputs are produced.

    python -m kws.stream --model models/dscnn_int8.onnx --wav stream.wav --threshold 0.8 \
        --smooth 3 --infer-every 5 --threads 1 --mode optimised \
        --detections out.csv --timings out.json
"""

import argparse
import csv
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from features import mfcc_ref
from kws.detector import NO_DETECTION, Detector
from kws.envinfo import write_results
from kws.ort_model import ORTModel
from kws.ring_buffer import FeatureRingBuffer
from kws.stats import WARMUP_INFERENCES, StageTimes, peak_rss_bytes
from kws.streaming_mfcc import StreamingMFCC

CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_unknown_", "_silence_",
]  # fmt: skip
SAMPLE_RATE = 16000


@dataclass
class StreamResult:
    detections: list[tuple[int, int, float]]  # (time in samples, class index, smoothed score)
    times: StageTimes
    logits: NDArray[np.float32] | None  # (n_inferences, 12) when record_logits
    inference_times: NDArray[np.int64] | None  # stream time of each inference, in samples
    alloc_bytes: NDArray[np.int64] | None = None  # per-hop Python allocation high-water mark


def run_stream(
    samples: NDArray[np.int16],
    model: ORTModel,
    detector: Detector,
    infer_every: int = 5,
    record_logits: bool = False,
    gc_enabled: bool = True,
    realtime: bool = False,
    mode: str = "optimised",
    alloc_hops: int = 0,
) -> StreamResult:
    """Run the streaming pipeline (`optimised` or `baseline`) over a whole int16 stream.

    With `alloc_hops` > 0, tracemalloc is started after the warm-up inferences and, for that many
    hops, records how far Python's traced memory rose above its level at the start of the hop
    (the high-water mark, so short-lived temporaries count). Tracing slows the loop, so a run
    with `alloc_hops` set is never used for timings.
    """
    if mode not in ("optimised", "baseline"):
        raise ValueError(f"unknown mode {mode!r}")
    baseline = mode == "baseline"
    spec = mfcc_ref.load_spec()
    cfg = spec.cfg
    hop = cfg["hop"]
    mean, std = mfcc_ref.norm_stats(spec)
    mfcc = StreamingMFCC(spec)
    features = FeatureRingBuffer(cfg["n_frames"], cfg["n_mfcc"], mean, std)
    frame = np.zeros(cfg["n_mfcc"], dtype=np.float64)
    clip_len = (cfg["n_frames"] - 1) * hop + cfg["window"]  # 16,000 samples = 49 frames
    input_name = model.session.get_inputs()[0].name

    n_hops = len(samples) // hop
    max_inf = n_hops // infer_every + 1
    times = StageTimes(n_hops)
    ns = times.ns
    ran = times.ran_inference
    logits_log = (
        np.zeros((max_inf, model.logits.shape[1]), dtype=np.float32) if record_logits else None
    )
    when = np.zeros(max_inf, dtype=np.int64)
    n_inf = 0
    detections: list[tuple[int, int, float]] = []
    clock = time.perf_counter_ns
    hop_ns = hop * 1_000_000_000 // SAMPLE_RATE

    alloc = np.zeros(alloc_hops, dtype=np.int64) if alloc_hops else None
    traced = 0  # hops traced so far
    get_traced = tracemalloc.get_traced_memory

    if not gc_enabled:
        gc.disable()
    t_start = clock()
    try:
        for h in range(n_hops):
            tracing = alloc is not None and n_inf >= WARMUP_INFERENCES and traced < alloc_hops
            if tracing:
                if traced == 0:
                    tracemalloc.start()
                base = get_traced()[0]
                tracemalloc.reset_peak()
            now = (h + 1) * hop
            t0 = clock()
            if not baseline:
                if mfcc.push(samples[h * hop : (h + 1) * hop], frame):
                    features.push(frame)
            t1 = clock()
            ns[0, h] = t1 - t0
            if (h + 1) % infer_every == 0 and now >= clip_len:
                if baseline:
                    # Naive path: every frame again, new arrays everywhere, plain session.run.
                    clip = samples[now - clip_len : now].astype(np.float64) * (1.0 / 32768.0)
                    norm = mfcc_ref.normalize(mfcc_ref.mfcc(clip, spec), spec)
                    x = norm.astype(np.float32).reshape(1, 1, cfg["n_frames"], cfg["n_mfcc"])
                    t1 = clock()
                    ns[0, h] = t1 - t0
                    logits = model.session.run(None, {input_name: x})[0][0]
                else:
                    features.read_into(model.frames)
                    logits = model.run()
                t2 = clock()
                cls = detector.update(logits, now)
                if cls != NO_DETECTION:
                    detections.append((now, cls, detector.score))
                t3 = clock()
                ns[1, h] = t2 - t1
                ns[2, h] = t3 - t2
                ran[h] = True
                if logits_log is not None:
                    logits_log[n_inf] = logits
                when[n_inf] = now
                n_inf += 1
            if tracing and alloc is not None:
                alloc[traced] = get_traced()[1] - base
                traced += 1
                if traced == alloc_hops:
                    tracemalloc.stop()
            if realtime:
                wait = t_start + (h + 1) * hop_ns - clock()
                if wait > 0:
                    time.sleep(wait / 1e9)
    finally:
        if not gc_enabled:
            gc.enable()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    return StreamResult(
        detections,
        times,
        logits_log[:n_inf] if logits_log is not None else None,
        when[:n_inf].copy() if record_logits else None,
        alloc[:traced] if alloc is not None else None,
    )


def alloc_summary(alloc: NDArray[np.int64] | None) -> dict[str, float | int] | None:
    """Mean/p99/max of the per-hop Python allocation high-water mark (bytes), or None."""
    if alloc is None or len(alloc) == 0:
        return None
    return {
        "hops_traced": len(alloc),
        "mean_bytes_per_hop": float(alloc.mean()),
        "p99_bytes_per_hop": float(np.percentile(alloc, 99)),
        "max_bytes_per_hop": int(np.max(alloc)),
    }


def pin_to_cpu(cpu: int) -> None:
    """Pin this process to one logical CPU (hybrid CPUs otherwise mix P and E cores)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1 << cpu):
            raise OSError("SetProcessAffinityMask failed")
    elif sys.platform.startswith("linux"):
        import os

        os.sched_setaffinity(0, {cpu})
    else:
        raise OSError("CPU pinning is not supported on this platform")


def read_wav(path: Path) -> NDArray[np.int16]:
    data, sr = sf.read(path, dtype="int16")
    if sr != SAMPLE_RATE or data.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got {sr} Hz, shape {data.shape}")
    samples: NDArray[np.int16] = data
    return samples


def write_detections(path: Path, detections: list[tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "label", "score"])
        for t, c, s in detections:
            w.writerow([f"{t / SAMPLE_RATE:.3f}", CLASSES[c], f"{s:.4f}"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--smooth", type=int, default=3)
    p.add_argument("--infer-every", type=int, default=5)
    p.add_argument("--threads", type=int, default=1, help="intra-op threads; 0 = all cores")
    p.add_argument("--mode", choices=["baseline", "optimised"], default="optimised")
    p.add_argument("--no-gc", action="store_true", help="disable the garbage collector in the loop")
    p.add_argument("--realtime", action="store_true", help="sleep so hops arrive every 20 ms")
    p.add_argument(
        "--alloc-hops", type=int, default=0,
        help="trace Python allocations over this many hops after warm-up (slow; no timings)",
    )  # fmt: skip
    p.add_argument("--pin-cpu", type=int, default=None, help="pin the process to this logical CPU")
    p.add_argument("--detections", type=Path)
    p.add_argument("--timings", type=Path)
    args = p.parse_args()

    if args.pin_cpu is not None:
        pin_to_cpu(args.pin_cpu)
    samples = read_wav(args.wav)
    model = ORTModel(args.model, args.threads)
    result = run_stream(
        samples,
        model,
        Detector(args.threshold, args.smooth),
        args.infer_every,
        gc_enabled=not args.no_gc,
        realtime=args.realtime,
        mode=args.mode,
        alloc_hops=args.alloc_hops,
    )
    for t, c, s in result.detections:
        print(f"{t / SAMPLE_RATE:9.2f} s  {CLASSES[c]:<6} {s:.3f}")
    print(f"{len(result.detections)} detections in {len(samples) / SAMPLE_RATE:.1f} s of audio")
    if args.detections:
        write_detections(args.detections, result.detections)
    if args.timings:
        write_results(
            args.timings,
            {
                "mode": args.mode,
                "model": args.model.name,
                "model_bytes": args.model.stat().st_size,
                "stream_seconds": len(samples) / SAMPLE_RATE,
                "settings": {
                    "threshold": args.threshold,
                    "smooth": args.smooth,
                    "infer_every": args.infer_every,
                    "gc_enabled": not args.no_gc,
                    "realtime": args.realtime,
                    "pinned_cpu": args.pin_cpu,
                },
                "peak_rss_bytes": peak_rss_bytes(),
                "deadline_misses_note": "a miss is a hop whose total processing time exceeds 20 ms",
                "timings": result.times.summary(),
                "allocations": alloc_summary(result.alloc_bytes),
                "note": "Python-on-host-CPU timings; not device or accelerator latency",
            },
            ort_threads={
                "intra_op_num_threads": args.threads,
                "inter_op_num_threads": 1,
            },
        )


if __name__ == "__main__":
    main()
