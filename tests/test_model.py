import pytest

torch = pytest.importorskip("torch")

from train.model import DSCNN, count_parameters  # noqa: E402


def test_output_shape() -> None:
    model = DSCNN().eval()
    assert model(torch.zeros(3, 1, 49, 10)).shape == (3, 12)


def test_parameter_count_is_what_the_spec_architecture_implies() -> None:
    ch = 64
    stem = 1 * ch * 10 * 4 + ch + 2 * ch  # conv + bias + BN gamma/beta
    block = (ch * 9 + ch + 2 * ch) + (ch * ch + ch + 2 * ch)  # depthwise + pointwise
    fc = ch * 12 + 12
    assert count_parameters(DSCNN()) == stem + 4 * block + fc
