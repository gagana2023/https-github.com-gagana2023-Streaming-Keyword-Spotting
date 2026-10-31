"""Bundle the KWS models and test features for Project 1 (its models/kws/ directory).

Writes to export/kws/ (gitignored; copy the folder into Project 1):
  dscnn_int8.onnx, dscnn_fp32.onnx   the models (INT8 is the one Project 1 must match)
  kws_test_features.npz              x (N,1,49,10) float32 normalised as at inference, y (N,)
                                     labels, plus ORT reference logits for both models
  SOURCE.json                        provenance: commit, versions, quantisation, accuracy

    python -m tools.export_for_project1
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from kws.envinfo import REPO_ROOT, collect_env
from train.data import CLASSES, load_dataset
from train.eval_onnx import INPUT_NAME, make_session, ort_features

MODELS = REPO_ROOT / "models"
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "export" / "kws"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_logits(model: Path, x: np.ndarray) -> np.ndarray:
    """Batch-1 ORT logits per clip, exactly as train.eval_onnx and the stream loop run them."""
    sess = make_session(model)
    out = np.empty((len(x), len(CLASSES)), dtype=np.float32)
    for i in range(len(x)):
        out[i] = sess.run(None, {INPUT_NAME: x[i : i + 1]})[0][0]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("dscnn_int8.onnx", "dscnn_fp32.onnx"):
        shutil.copy2(MODELS / name, OUT / name)

    data = load_dataset().splits["test"]
    x = ort_features(data.x)
    y = data.y.astype(np.int64)
    logits_int8 = reference_logits(OUT / "dscnn_int8.onnx", x)
    logits_fp32 = reference_logits(OUT / "dscnn_fp32.onnx", x)
    np.savez(
        OUT / "kws_test_features.npz",
        x=x,
        y=y,
        logits_int8=logits_int8,
        logits_fp32=logits_fp32,
    )

    acc: dict[str, Any] = json.loads((RESULTS / "accuracy.json").read_text())
    quant: dict[str, Any] = json.loads((RESULTS / "quantization.json").read_text())
    q_env = quant["env"]
    int8_acc = float((logits_int8.argmax(1) == y).mean())
    fp32_acc = float((logits_fp32.argmax(1) == y).mean())
    # The exported reference logits must reproduce the committed accuracy numbers exactly.
    assert abs(int8_acc - acc["int8_accuracy"]) < 1e-12, (int8_acc, acc["int8_accuracy"])
    assert abs(fp32_acc - acc["fp32_accuracy"]) < 1e-12, (fp32_acc, acc["fp32_accuracy"])

    env = collect_env()
    source = {
        "project": "Project 2 - Streaming Keyword Spotting",
        "project2_git_commit": env["git_commit"],
        "project2_git_dirty": env["git_dirty"],
        "exported_utc": env["date_utc"],
        "quantised_at_utc": q_env["date_utc"],
        "quantised_with": {
            "onnx": q_env["packages"]["onnx"],
            "onnxruntime": q_env["packages"]["onnxruntime"],
            "python": q_env["python"],
        },
        "quantisation": quant["settings"],
        "test_accuracy": {
            "split": "test",
            "clips": acc["test_clips"],
            "fp32": acc["fp32_accuracy"],
            "int8": acc["int8_accuracy"],
            "source": "results/accuracy.json",
        },
        "model": {
            "input_name": INPUT_NAME,
            "input_shape": [1, 1, 49, 10],
            "classes": CLASSES,
            "parameter_count": acc["parameter_count"],
        },
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(OUT.glob("*"))
            if p.suffix in (".onnx", ".npz")
        },
        "npz": {
            "x": "float32 (N,1,49,10), normalised as at inference (train.eval_onnx.ort_features)",
            "y": "int64 (N,), index into model.classes",
            "logits_int8": "float32 (N,12), ORT CPU batch-1 reference output of dscnn_int8.onnx",
            "logits_fp32": "float32 (N,12), ORT CPU batch-1 reference output of dscnn_fp32.onnx",
        },
        "export_command": env["command"],
    }
    (OUT / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"wrote {OUT}: N={len(y)}, INT8 acc {int8_acc:.4f}, FP32 acc {fp32_acc:.4f}")


if __name__ == "__main__":
    main()
