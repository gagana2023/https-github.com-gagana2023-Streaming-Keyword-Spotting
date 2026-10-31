"""M0 acceptance: a tiny ONNX model loads and runs through ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def _build_tiny_model(path: Path) -> np.ndarray:
    """y = relu(x @ W + b), x: 1x4, y: 1x3. Returns W so the test can check the output."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    b = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "W", "b"], ["z"]),
            helper.make_node("Relu", ["z"], ["y"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return w


def test_tiny_model_runs_through_ort(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    w = _build_tiny_model(path)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
    (y,) = sess.run(None, {"x": x})

    expected = np.maximum(x @ w + np.array([0.5, -0.5, 0.0], dtype=np.float32), 0)
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)
