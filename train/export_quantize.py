"""Export the trained DS-CNN to ONNX (opset 17) and quantise it to INT8 with ORT.

Pipeline (same as Project 1): torch.onnx.export -> ORT quant_pre_process -> quantize_static
with QOperator format, QInt8 activations and weights, per-channel weights, and calibration
clips drawn from the training split.

Run: python -m train.export_quantize [--calibration minmax|entropy|percentile] ...
Each run is appended to results/quantization_attempts.json (accuracy on validation only) and
the settings of the latest run go to results/quantization.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from numpy.typing import NDArray
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from kws.envinfo import write_results
from train.data import SEED, load_dataset
from train.eval_onnx import (
    FP32_PATH,
    INPUT_NAME,
    INT8_PATH,
    MODELS_DIR,
    RESULTS_DIR,
    accuracy,
    make_session,
    ort_features,
)
from train.model import DSCNN

OPSET = 17
METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class ClipReader(CalibrationDataReader):  # type: ignore[misc]
    """Feeds one calibration clip (1, 1, 49, 10) per call."""

    def __init__(self, feats: NDArray[np.float32]) -> None:
        self._it = iter([feats[i : i + 1] for i in range(len(feats))])

    def get_next(self) -> dict[str, Any] | None:
        x = next(self._it, None)
        return None if x is None else {INPUT_NAME: x}


def export_fp32(checkpoint: Path, out: Path) -> DSCNN:
    model = DSCNN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(1, 1, 49, 10),),
        str(out),
        input_names=[INPUT_NAME],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(out)))
    return model


def quantize_model(
    fp32_path: Path,
    int8_path: Path,
    calib: NDArray[np.float32],
    per_channel: bool = True,
    method: str = "minmax",
) -> None:
    """ORT pre-process, then static INT8 quantisation in QOperator format."""
    pre_path = fp32_path.with_name(fp32_path.stem + "_preprocessed.onnx")
    quant_pre_process(str(fp32_path), str(pre_path))
    quantize_static(
        str(pre_path),
        str(int8_path),
        ClipReader(calib),
        quant_format=QuantFormat.QOperator,
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=METHODS[method],
    )
    onnx.checker.check_model(onnx.load(str(int8_path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "dscnn.pt")
    parser.add_argument("--calibration", choices=sorted(METHODS), default="minmax")
    parser.add_argument("--num-calib", type=int, default=200)
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    model = export_fp32(args.checkpoint, FP32_PATH)

    ds = load_dataset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    calib_idx = np.sort(rng.choice(len(ds.splits["train"].y), args.num_calib, replace=False))
    calib = ort_features(ds.splits["train"].x[calib_idx])
    quantize_model(FP32_PATH, INT8_PATH, calib, args.per_channel, args.calibration)

    # Sanity: the FP32 ONNX graph must reproduce the PyTorch logits.
    val = ds.splits["val"]
    feats = ort_features(val.x)
    sess = make_session(FP32_PATH)
    probe = feats[:200]
    ort_logits = np.concatenate(
        [sess.run(None, {INPUT_NAME: probe[i : i + 1]})[0] for i in range(len(probe))]
    )
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(probe)).numpy()
    max_diff = float(np.abs(ort_logits - torch_logits).max())
    print(f"max |torch - onnx fp32| logit difference: {max_diff:.2e}")
    assert max_diff < 1e-3

    fp32_val, _ = accuracy(FP32_PATH, feats, val.y)
    int8_val, _ = accuracy(INT8_PATH, feats, val.y)
    print(f"validation: FP32 {fp32_val:.4f}  INT8 {int8_val:.4f}  drop {fp32_val - int8_val:.4f}")

    settings = {
        "format": "QOperator",
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "per_channel": args.per_channel,
        "calibration_method": args.calibration,
        "calibration_clips": args.num_calib,
        "calibration_source": "random clips from the training split, unaugmented",
        "opset": OPSET,
        "preprocess": "onnxruntime.quantization.shape_inference.quant_pre_process",
    }
    result = {
        "settings": settings,
        "val_accuracy_fp32": fp32_val,
        "val_accuracy_int8": int8_val,
        "torch_vs_onnx_fp32_max_logit_diff": max_diff,
    }
    write_results(RESULTS_DIR / "quantization.json", result)
    attempts_path = RESULTS_DIR / "quantization_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    attempts.append(result)
    attempts_path.write_text(json.dumps(attempts, indent=2) + "\n")


if __name__ == "__main__":
    main()
