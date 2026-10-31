"""Sweep threshold and smoothing window on the VALIDATION stream and choose the operating point.

Run: python -m tools.sweep_threshold [--max-fa-per-hour 10]

The model is run over the stream once; the logits are recorded and replayed through the same
Detector for every (smoothing window, threshold) pair, because neither changes the features or
the model output. Writes results/stream_eval_val.json (whole grid + chosen point) and
results/threshold_sweep.png. The test stream is never read here.

Selection rule: among grid points with at most `--max-fa-per-hour` false accepts per hour on
the validation stream, take the lowest miss rate (ties: fewer false accepts, then the higher
threshold, then the smaller window). If no point qualifies, take the fewest false accepts.
"""

import argparse
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from kws.detector import Detector, detect_from_logits  # noqa: E402
from kws.envinfo import REPO_ROOT, write_results  # noqa: E402
from kws.ort_model import ORTModel  # noqa: E402
from kws.stream import SAMPLE_RATE, read_wav, run_stream  # noqa: E402
from tools.eval_stream import MODELS, detections_to_named, score  # noqa: E402
from tools.make_streams import read_events, stream_paths  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
SMOOTH_GRID = (1, 2, 3, 4, 5)
THRESHOLD_GRID = tuple(
    sorted(
        {round(float(t), 3) for t in np.arange(0.30, 0.90, 0.025)}
        | {round(float(t), 3) for t in np.arange(0.90, 0.99, 0.01)}
        | {0.99, 0.995, 0.999}
    )
)
DEFAULT_MAX_FA_PER_HOUR = 10.0


def choose(rows: Sequence[dict[str, Any]], max_fa_per_hour: float) -> dict[str, Any]:
    ok = [r for r in rows if r["false_accepts_per_hour"] <= max_fa_per_hour]
    if not ok:
        return min(rows, key=lambda r: (r["false_accepts_per_hour"], r["miss_rate"]))
    return min(
        ok,
        key=lambda r: (r["miss_rate"], r["false_accepts_per_hour"], -r["threshold"], r["smooth"]),
    )


def plot(rows: Sequence[dict[str, Any]], chosen: dict[str, Any], path: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in SMOOTH_GRID:
        pts = sorted(
            (r for r in rows if r["smooth"] == k), key=lambda r: r["false_accepts_per_hour"]
        )
        ax.plot(
            [r["false_accepts_per_hour"] for r in pts],
            [100 * r["miss_rate"] for r in pts],
            marker="o", markersize=3, linewidth=1, label=f"smooth k={k}",
        )  # fmt: skip
    ax.plot(
        chosen["false_accepts_per_hour"], 100 * chosen["miss_rate"], "k*", markersize=14,
        label=f"chosen: k={chosen['smooth']}, thr={chosen['threshold']}",
    )  # fmt: skip
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("false accepts per hour (validation stream, symlog)")
    ax.set_ylabel("miss rate (%)")
    ax.set_title("Threshold sweep, INT8 model, validation stream")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-fa-per-hour", type=float, default=DEFAULT_MAX_FA_PER_HOUR)
    p.add_argument("--infer-every", type=int, default=5)
    args = p.parse_args()

    wav_path, csv_path = stream_paths("val")
    samples = read_wav(wav_path)
    events = read_events(csv_path)
    duration_s = len(samples) / SAMPLE_RATE

    model = ORTModel(MODELS["int8"], threads=1)
    result = run_stream(samples, model, Detector(0.5), args.infer_every, record_logits=True)
    assert result.logits is not None and result.inference_times is not None

    rows: list[dict[str, Any]] = []
    for k in SMOOTH_GRID:
        for thr in THRESHOLD_GRID:
            dets = detect_from_logits(result.logits, result.inference_times, thr, k)
            m = score(events, detections_to_named([(t, c, s) for t, c, s in dets]), duration_s)
            rows.append(
                {
                    "smooth": k,
                    "threshold": thr,
                    "hits": m["hits"],
                    "misses": m["misses"],
                    "miss_rate": m["miss_rate"],
                    "false_accepts": m["false_accepts"],
                    "false_accepts_per_hour": m["false_accepts_per_hour"],
                }
            )
    chosen = choose(rows, args.max_fa_per_hour)
    chosen = {
        **chosen,
        "rule": (
            f"lowest miss rate with <= {args.max_fa_per_hour} false accepts/hour on the "
            "validation stream; ties: fewer false accepts, higher threshold, smaller window"
        ),
    }
    print("chosen:", chosen)

    plot(rows, chosen, RESULTS_DIR / "threshold_sweep.png")
    write_results(
        RESULTS_DIR / "stream_eval_val.json",
        {
            "split": "val",
            "model": MODELS["int8"].name,
            "stream_seconds": duration_s,
            "keywords": sum(e.label != "_unknown_" for e in events),
            "infer_every": args.infer_every,
            "suppress_s": 1.0,
            "max_fa_per_hour": args.max_fa_per_hour,
            "chosen": chosen,
            "grid": rows,
        },
        ort_threads={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    )


if __name__ == "__main__":
    main()
