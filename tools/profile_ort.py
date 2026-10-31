"""Per-operator ORT profile of the FP32 and INT8 models (explains the INT8 vs FP32 timing).

Runs each model for `--runs` inferences on a fixed random input with ORT's built-in profiler
(1 intra-op thread), sums kernel time per operator type, and splits convolutions into depthwise
(group > 1) and pointwise/dense (group == 1) using the ONNX graph. ORT renames fused FP32
Conv+Relu nodes, so those cannot be matched and are reported as `unattributed`.

    python -m tools.profile_ort

Writes results/ort_profile.json. Profiling adds overhead, so use the ratios, not the absolute
times, and compare with the latency_*.json files for real per-inference time.
"""

import argparse
import collections
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from kws.envinfo import REPO_ROOT, write_results

MODELS = {"fp32": REPO_ROOT / "models" / "dscnn_fp32.onnx",
          "int8": REPO_ROOT / "models" / "dscnn_int8.onnx"}  # fmt: skip
CONV_OPS = ("Conv", "QLinearConv")


def conv_groups(path: Path) -> dict[str, int]:
    """Node name -> group attribute for every convolution in the model."""
    groups: dict[str, int] = {}
    for node in onnx.load(str(path)).graph.node:
        if node.op_type in CONV_OPS:
            groups[node.name] = next((a.i for a in node.attribute if a.name == "group"), 1)
    return groups


def profile(path: Path, runs: int, tmp: Path) -> dict[str, Any]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    opts.profile_file_prefix = str(tmp / path.stem)
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((1, 1, 49, 10)).astype(np.float32)
    for _ in range(runs):
        session.run(None, {name: x})
    events = json.loads(Path(session.end_profiling()).read_text())

    groups = conv_groups(path)
    per_op: dict[str, float] = collections.defaultdict(float)
    conv_split = {"depthwise": 0.0, "pointwise_or_dense": 0.0, "unattributed": 0.0}
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        us = e["dur"] / runs
        op = e["args"]["op_name"]
        per_op[op] += us
        if op in CONV_OPS:
            node = re.sub(r"_token_\d+$", "", e["name"].removesuffix("_kernel_time"))
            if node not in groups:  # ORT renamed a fused node (FP32 Conv+Relu); can't attribute
                conv_split["unattributed"] += us
                continue
            kind = "depthwise" if groups[node] > 1 else "pointwise_or_dense"
            conv_split[kind] += us
    total = sum(per_op.values())
    return {
        "kernel_us_per_inference_total": total,
        "per_op_us": dict(sorted(per_op.items(), key=lambda kv: -kv[1])),
        "conv_us_by_kind": conv_split,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--runs", type=int, default=500)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        out = {name: profile(path, args.runs, Path(tmp)) for name, path in MODELS.items()}
    ratio = (
        out["int8"]["kernel_us_per_inference_total"] / out["fp32"]["kernel_us_per_inference_total"]
    )
    for name, r in out.items():
        print(name, round(r["kernel_us_per_inference_total"], 1), "us", r["conv_us_by_kind"])
    write_results(
        REPO_ROOT / "results" / "ort_profile.json",
        {
            "runs": args.runs,
            "models": out,
            "int8_over_fp32_kernel_time": ratio,
            "note": "ORT profiler, 1 intra-op thread, Python on host CPU; profiling adds overhead",
        },
        ort_threads={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    )


if __name__ == "__main__":
    main()
