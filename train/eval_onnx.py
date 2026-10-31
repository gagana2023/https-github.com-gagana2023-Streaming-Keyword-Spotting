"""Evaluate the FP32 and INT8 ONNX models with ONNX Runtime on a split.

Run: python -m train.eval_onnx             (test split, writes results/accuracy.json)
     python -m train.eval_onnx --split val  (validation, prints only; for tuning)
The test split is for final numbers only (spec honesty rule 3).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from features import mfcc_ref
from kws.envinfo import REPO_ROOT, write_results
from train.data import CLASSES, SplitData, class_counts, load_dataset

MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FP32_PATH = MODELS_DIR / "dscnn_fp32.onnx"
INT8_PATH = MODELS_DIR / "dscnn_int8.onnx"
INPUT_NAME = "mfcc"


def make_session(path: Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_features(pcm: NDArray[np.int16], chunk: int = 1000) -> NDArray[np.float32]:
    """int16 clips (N, 16000) -> normalised float32 features (N, 1, 49, 10) via the reference."""
    out = np.empty((len(pcm), 1, 49, 10), dtype=np.float32)
    for i in range(0, len(pcm), chunk):
        m = mfcc_ref.mfcc(pcm[i : i + chunk].astype(np.float64) / 32768.0)
        out[i : i + chunk, 0] = mfcc_ref.normalize(m)
    return out


def predict(sess: ort.InferenceSession, feats: NDArray[np.float32]) -> NDArray[np.int64]:
    """Batch-1 inference per clip, exactly as the streaming pipeline will run it."""
    preds = np.empty(len(feats), dtype=np.int64)
    for i in range(len(feats)):
        (logits,) = sess.run(None, {INPUT_NAME: feats[i : i + 1]})
        preds[i] = int(np.argmax(logits))
    return preds


def accuracy(
    path: Path, feats: NDArray[np.float32], y: NDArray[np.int64], threads: int = 1
) -> tuple[float, NDArray[np.int64]]:
    preds = predict(make_session(path, threads), feats)
    return float((preds == y).mean()), preds


def per_class_accuracy(preds: NDArray[np.int64], y: NDArray[np.int64]) -> dict[str, float]:
    return {name: float((preds[y == i] == i).mean()) for i, name in enumerate(CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    data: SplitData = load_dataset().splits[args.split]
    feats = ort_features(data.x)
    fp32_acc, fp32_pred = accuracy(FP32_PATH, feats, data.y)
    int8_acc, int8_pred = accuracy(INT8_PATH, feats, data.y)
    delta = int8_acc - fp32_acc
    print(f"{args.split}: FP32 {fp32_acc:.4f}  INT8 {int8_acc:.4f}  delta {delta:+.4f}")
    if args.split != "test":
        return

    train_summary = json.loads((RESULTS_DIR / "train_summary.json").read_text())
    quant: dict[str, Any] = json.loads((RESULTS_DIR / "quantization.json").read_text())
    write_results(
        RESULTS_DIR / "accuracy.json",
        {
            "split": "test",
            "test_clips": int(len(data.y)),
            "test_class_counts": class_counts(data.y),
            "fp32_accuracy": fp32_acc,
            "int8_accuracy": int8_acc,
            "int8_minus_fp32": delta,
            "fp32_per_class_accuracy": per_class_accuracy(fp32_pred, data.y),
            "int8_per_class_accuracy": per_class_accuracy(int8_pred, data.y),
            "predictions_differing_fp32_vs_int8": int((fp32_pred != int8_pred).sum()),
            "parameter_count": train_summary["parameter_count"],
            "model_size_bytes": {
                "fp32_onnx": FP32_PATH.stat().st_size,
                "int8_onnx": INT8_PATH.stat().st_size,
            },
            "quantization": quant["settings"],
            "features": "NumPy reference (features/mfcc_ref.py), normalised, float32 into ORT",
            "ort_session": "batch 1, CPUExecutionProvider",
        },
        ort_threads={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    )


if __name__ == "__main__":
    main()
