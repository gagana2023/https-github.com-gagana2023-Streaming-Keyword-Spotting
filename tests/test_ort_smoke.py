"""M0 acceptance: a tiny ONNX model loads and runs through ONNX Runtime."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def _build_tiny_model(path: Path) -> None:
    """y = relu(x @ W + b) with W = 2*I (4x4) and b = -1."""
    w = helper.make_tensor("W", TensorProto.FLOAT, [4, 4], (2 * np.eye(4)).flatten().tolist())
    b = helper.make_tensor("b", TensorProto.FLOAT, [4], [-1.0] * 4)
    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["xw"]),
        helper.make_node("Add", ["xw", "b"], ["z"]),
        helper.make_node("Relu", ["z"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_ort_loads_and_runs_tiny_model(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    _build_tiny_model(path)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.array([[0.0, 0.5, 1.0, 2.0]], dtype=np.float32)
    (y,) = sess.run(None, {"x": x})

    np.testing.assert_allclose(y, [[0.0, 0.0, 1.0, 3.0]], atol=1e-6)
