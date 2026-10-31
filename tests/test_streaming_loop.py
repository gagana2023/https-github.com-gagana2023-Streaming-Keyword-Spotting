"""End-to-end checks of ORTModel and run_stream on a tiny synthetic ONNX model (CI-safe)."""

import tracemalloc
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from kws.detector import Detector, detect_from_logits
from kws.ort_model import ORTModel
from kws.stats import peak_rss_bytes
from kws.stream import run_stream


def make_tiny_model(path: Path) -> None:
    """logits = flatten(mfcc) @ W, with a fixed random W: (1,1,49,10) -> (1,12)."""
    w = np.random.default_rng(0).normal(0, 0.05, size=(490, 12)).astype(np.float32)
    nodes = [
        helper.make_node("Flatten", ["mfcc"], ["flat"], axis=1),
        helper.make_node("MatMul", ["flat", "W"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_kws",
        [helper.make_tensor_value_info("mfcc", TensorProto.FLOAT, [1, 1, 49, 10])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 12])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, [490, 12], w.flatten().tolist())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_iobinding_matches_session_run_after_in_place_updates(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    make_tiny_model(path)
    model = ORTModel(path)
    rng = np.random.default_rng(1)
    for _ in range(3):  # the bound buffers must see each new in-place write
        model.frames[:] = rng.standard_normal((49, 10)).astype(np.float32)
        got = model.run().copy()
        (ref,) = model.session.run(None, {"mfcc": model.input.copy()})
        np.testing.assert_allclose(got, ref[0], atol=1e-6)


def test_run_stream_schedule_and_replay(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    make_tiny_model(path)
    rng = np.random.default_rng(2)
    samples = (rng.standard_normal(16000 * 6) * 3000).astype(np.int16)  # 6 s of noise
    model = ORTModel(path)
    res = run_stream(samples, model, Detector(0.2, 3), infer_every=5, record_logits=True)

    # 300 hops; the first inference is at hop 50 (49 frames exist), then every 5 hops.
    assert res.inference_times is not None and res.logits is not None
    assert res.inference_times[0] == 50 * 320
    assert len(res.inference_times) == (300 - 50) // 5 + 1
    assert np.all(np.diff(res.inference_times) == 5 * 320)
    # Replaying the recorded logits gives exactly the live detections.
    assert detect_from_logits(res.logits, res.inference_times, 0.2, 3) == res.detections
    summary = res.times.summary(warmup_inferences=10)
    assert summary["inferences_measured"] == len(res.inference_times) - 10


def test_baseline_matches_optimised_logits(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    make_tiny_model(path)
    rng = np.random.default_rng(3)
    samples = (rng.standard_normal(16000 * 4) * 3000).astype(np.int16)
    fast = run_stream(samples, ORTModel(path), Detector(0.2, 3), record_logits=True)
    slow = run_stream(
        samples, ORTModel(path), Detector(0.2, 3), record_logits=True, mode="baseline"
    )
    assert fast.logits is not None and slow.logits is not None
    assert fast.inference_times is not None and slow.inference_times is not None
    np.testing.assert_array_equal(fast.inference_times, slow.inference_times)
    np.testing.assert_allclose(fast.logits, slow.logits, atol=1e-4)
    assert fast.detections == slow.detections


def test_allocation_probe_and_peak_rss(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    make_tiny_model(path)
    rng = np.random.default_rng(4)
    samples = (rng.standard_normal(16000 * 12) * 3000).astype(np.int16)
    res = run_stream(samples, ORTModel(path), Detector(0.99, 3), alloc_hops=100)
    assert res.alloc_bytes is not None and len(res.alloc_bytes) == 100
    assert np.all(res.alloc_bytes >= 0)
    assert not tracemalloc.is_tracing()
    assert peak_rss_bytes() > 1_000_000
