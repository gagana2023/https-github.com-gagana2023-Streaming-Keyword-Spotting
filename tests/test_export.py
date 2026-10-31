from pathlib import Path

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch")

from train.eval_onnx import INPUT_NAME, make_session  # noqa: E402
from train.export_quantize import export_fp32, quantize_model  # noqa: E402
from train.model import DSCNN  # noqa: E402


def test_export_and_quantize_random_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    ckpt = tmp_path / "m.pt"
    torch.save(DSCNN().state_dict(), ckpt)
    fp32 = tmp_path / "fp32.onnx"
    model = export_fp32(ckpt, fp32)

    assert onnx.load(str(fp32)).opset_import[0].version == 17
    x = np.random.default_rng(0).standard_normal((20, 1, 49, 10)).astype(np.float32)
    sess = make_session(fp32)
    ort_out = np.concatenate([sess.run(None, {INPUT_NAME: x[i : i + 1]})[0] for i in range(20)])
    with torch.no_grad():
        np.testing.assert_allclose(ort_out, model(torch.from_numpy(x)).numpy(), atol=1e-4)

    int8 = tmp_path / "int8.onnx"
    quantize_model(fp32, int8, x)
    ops = {n.op_type for n in onnx.load(str(int8)).graph.node}
    assert "QLinearConv" in ops  # QOperator format, not QDQ
    q = make_session(int8).run(None, {INPUT_NAME: x[:1]})[0]
    assert q.shape == (1, 12) and np.isfinite(q).all()
