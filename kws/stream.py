"""Streaming keyword spotting loop (optimised mode).

A 16 kHz mono int16 WAV is fed in hops of 320 samples, as if it arrived live. Per hop:

  1. StreamingMFCC turns the newest 320 samples (plus the 320 held over) into one frame.
  2. The frame is normalised into the FeatureRingBuffer.
  3. Every `infer_every` hops, once 49 frames exist, the buffer is copied in time order into the
     ORT input array, the model runs through IOBinding, and the Detector sees the logits.

Everything is preallocated before the loop. Detection time is the stream time (samples consumed
so far) of the inference that fired; the 49 frames cover exactly the last 16,000 samples.

Baseline mode (recompute all frames per inference, fresh arrays, plain `session.run`) is added
in M4; only `optimised` exists now.

    python -m kws.stream --model models/dscnn_int8.onnx --wav stream.wav --threshold 0.8 \
        --smooth 3 --infer-every 5 --threads 1 --mode optimised \
        --detections out.csv --timings out.json
"""

import argparse
import csv
import gc
import time
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
from kws.stats import StageTimes
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


def run_stream(
    samples: NDArray[np.int16],
    model: ORTModel,
    detector: Detector,
    infer_every: int = 5,
    record_logits: bool = False,
    gc_enabled: bool = True,
    realtime: bool = False,
) -> StreamResult:
    """Run the optimised streaming pipeline over a whole int16 stream."""
    spec = mfcc_ref.load_spec()
    cfg = spec.cfg
    hop = cfg["hop"]
    mean, std = mfcc_ref.norm_stats(spec)
    mfcc = StreamingMFCC(spec)
    features = FeatureRingBuffer(cfg["n_frames"], cfg["n_mfcc"], mean, std)
    frame = np.zeros(cfg["n_mfcc"], dtype=np.float64)

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

    if not gc_enabled:
        gc.disable()
    t_start = clock()
    try:
        for h in range(n_hops):
            t0 = clock()
            if mfcc.push(samples[h * hop : (h + 1) * hop], frame):
                features.push(frame)
            t1 = clock()
            ns[0, h] = t1 - t0
            now = (h + 1) * hop
            if (h + 1) % infer_every == 0 and features.full:
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
            if realtime:
                wait = t_start + (h + 1) * hop_ns - clock()
                if wait > 0:
                    time.sleep(wait / 1e9)
    finally:
        if not gc_enabled:
            gc.enable()
    return StreamResult(
        detections,
        times,
        logits_log[:n_inf] if logits_log is not None else None,
        when[:n_inf].copy() if record_logits else None,
    )


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
    p.add_argument("--mode", choices=["optimised"], default="optimised")
    p.add_argument("--no-gc", action="store_true", help="disable the garbage collector in the loop")
    p.add_argument("--realtime", action="store_true", help="sleep so hops arrive every 20 ms")
    p.add_argument("--detections", type=Path)
    p.add_argument("--timings", type=Path)
    args = p.parse_args()

    samples = read_wav(args.wav)
    model = ORTModel(args.model, args.threads)
    result = run_stream(
        samples,
        model,
        Detector(args.threshold, args.smooth),
        args.infer_every,
        gc_enabled=not args.no_gc,
        realtime=args.realtime,
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
                },
                "timings": result.times.summary(),
                "note": "Python-on-host-CPU timings; not device or accelerator latency",
            },
            ort_threads={
                "intra_op_num_threads": args.threads,
                "inter_op_num_threads": 1,
            },
        )


if __name__ == "__main__":
    main()
