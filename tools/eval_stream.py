"""Score detections against a stream's ground truth, and the final test-stream evaluation.

Scoring rules (spec 3.6):
  * A hit is a detection with the correct label inside [start - 0.25 s, end + 0.75 s] of a
    keyword. Each keyword can be hit once, by its earliest such detection.
  * Miss rate = missed keywords / all keywords.
  * Every other detection is a false accept, including wrong labels and repeats. They are
    split into categories for diagnosis only; the headline number counts all of them.
  * False accepts per hour = false accepts / stream length in hours.

Run (final numbers; uses the operating point chosen on the validation stream):
    python -m tools.eval_stream
The threshold and smoothing window are read from results/stream_eval_val.json, so the test
stream cannot influence them.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kws.detector import Detector
from kws.envinfo import REPO_ROOT, write_results
from kws.ort_model import ORTModel
from kws.stream import CLASSES, SAMPLE_RATE, read_wav, run_stream, write_detections
from tools.make_streams import Event, read_events, stream_paths

HIT_BEFORE_S = 0.25
HIT_AFTER_S = 0.75
UNKNOWN = "_unknown_"
RESULTS_DIR = REPO_ROOT / "results"
MODELS = {"int8": REPO_ROOT / "models" / "dscnn_int8.onnx",
          "fp32": REPO_ROOT / "models" / "dscnn_fp32.onnx"}  # fmt: skip


def score(
    events: Sequence[Event],
    detections: Sequence[tuple[float, str]],
    duration_s: float,
) -> dict[str, Any]:
    """Match (time_s, label) detections to ground-truth events; return the metrics."""
    keywords = [e for e in events if e.label != UNKNOWN]
    unknowns = [e for e in events if e.label == UNKNOWN]
    hit_by: list[float | None] = [None] * len(keywords)
    false_accepts: list[tuple[float, str]] = []
    for t, label in sorted(detections):
        for i, e in enumerate(keywords):
            if (
                hit_by[i] is None
                and e.label == label
                and e.start_s - HIT_BEFORE_S <= t <= e.end_s + HIT_AFTER_S
            ):
                hit_by[i] = t
                break
        else:
            false_accepts.append((t, label))

    def inside(t: float, evs: Sequence[Event]) -> bool:
        return any(e.start_s - HIT_BEFORE_S <= t <= e.end_s + HIT_AFTER_S for e in evs)

    categories = {
        "wrong_or_repeat_in_keyword_window": 0,
        "in_unknown_word_window": 0,
        "elsewhere": 0,
    }
    for t, _ in false_accepts:
        if inside(t, keywords):
            categories["wrong_or_repeat_in_keyword_window"] += 1
        elif inside(t, unknowns):
            categories["in_unknown_word_window"] += 1
        else:
            categories["elsewhere"] += 1

    hits = sum(h is not None for h in hit_by)
    delays = [h - e.end_s for h, e in zip(hit_by, keywords, strict=True) if h is not None]
    hours = duration_s / 3600.0
    return {
        "keywords": len(keywords),
        "unknown_words": len(unknowns),
        "hits": hits,
        "misses": len(keywords) - hits,
        "miss_rate": (len(keywords) - hits) / len(keywords),
        "false_accepts": len(false_accepts),
        "false_accepts_per_hour": len(false_accepts) / hours,
        "false_accept_categories": categories,
        "stream_hours": hours,
        "median_detection_delay_after_word_end_s": sorted(delays)[len(delays) // 2]
        if delays
        else None,
    }


def detections_to_named(dets: Sequence[tuple[int, int, float]]) -> list[tuple[float, str]]:
    return [(t / SAMPLE_RATE, CLASSES[c]) for t, c, _ in dets]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--val-results", type=Path, default=RESULTS_DIR / "stream_eval_val.json")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "stream_eval_test.json")
    p.add_argument("--infer-every", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    chosen = json.loads(args.val_results.read_text())["chosen"]
    threshold, smooth = chosen["threshold"], chosen["smooth"]
    wav_path, csv_path = stream_paths("test")
    samples = read_wav(wav_path)
    events = read_events(csv_path)
    duration_s = len(samples) / SAMPLE_RATE

    runs: dict[str, Any] = {}
    for name, path in MODELS.items():  # INT8 is the deployed model; FP32 is for comparison
        result = run_stream(
            samples, ORTModel(path, args.threads), Detector(threshold, smooth), args.infer_every
        )
        runs[name] = score(events, detections_to_named(result.detections), duration_s)
        write_detections(RESULTS_DIR / f"stream_detections_test_{name}.csv", result.detections)
        print(
            name,
            json.dumps(
                {
                    k: runs[name][k]
                    for k in (
                        "hits",
                        "misses",
                        "miss_rate",
                        "false_accepts",
                        "false_accepts_per_hour",
                    )
                }
            ),
        )

    write_results(
        args.out,
        {
            "split": "test",
            "stream_seconds": duration_s,
            "operating_point": {
                "threshold": threshold,
                "smooth": smooth,
                "infer_every": args.infer_every,
                "suppress_s": 1.0,
                "chosen_on": "validation stream (results/stream_eval_val.json)",
                "selection_rule": chosen.get("rule"),
            },
            "scoring": {"hit_window_s": [-HIT_BEFORE_S, HIT_AFTER_S]},
            "primary_model": "int8",
            "models": runs,
        },
        ort_threads={"intra_op_num_threads": args.threads, "inter_op_num_threads": 1},
    )


if __name__ == "__main__":
    main()
