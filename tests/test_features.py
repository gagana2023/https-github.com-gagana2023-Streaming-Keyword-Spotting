import numpy as np
import pytest

from features.build_matrices import dct_matrix, hz_to_mel, mel_filterbank, mel_to_hz
from features.mfcc_ref import frame_count, hann_window, load_spec, mfcc


def test_config_matches_spec() -> None:
    cfg = load_spec().cfg
    assert (cfg["sample_rate"], cfg["window"], cfg["hop"], cfg["n_fft"]) == (16000, 640, 320, 1024)
    assert (cfg["n_mels"], cfg["fmin"], cfg["fmax"], cfg["n_mfcc"]) == (40, 20.0, 4000.0, 10)
    assert frame_count(cfg["clip_samples"], cfg["window"], cfg["hop"]) == cfg["n_frames"] == 49


def test_mel_scale_roundtrip() -> None:
    f = np.array([20.0, 440.0, 4000.0])
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(f)), f)


def test_mel_filterbank_properties() -> None:
    fb = mel_filterbank(16000, 1024, 40, 20.0, 4000.0)
    assert fb.shape == (40, 513)
    assert fb.min() >= 0.0 and fb.max() <= 1.0
    assert (fb.sum(axis=1) > 0).all()  # no empty band
    bin_hz = np.arange(513) * 16000 / 1024
    assert fb[:, bin_hz < 20.0].sum() == 0 and fb[:, bin_hz > 4000.0].sum() == 0
    # Band centres increase monotonically.
    centres = (fb * bin_hz).sum(axis=1) / fb.sum(axis=1)
    assert (np.diff(centres) > 0).all()


def test_committed_matrices_match_builder() -> None:
    spec = load_spec()
    cfg = spec.cfg
    mel = mel_filterbank(cfg["sample_rate"], cfg["n_fft"], cfg["n_mels"], cfg["fmin"], cfg["fmax"])
    np.testing.assert_array_equal(spec.mel, mel)
    np.testing.assert_array_equal(spec.dct, dct_matrix(cfg["n_mels"], cfg["n_mfcc"]))


def test_dct_orthonormal_and_matches_definition() -> None:
    d = dct_matrix(40, 40)
    np.testing.assert_allclose(d @ d.T, np.eye(40), atol=1e-12)
    x = np.random.default_rng(0).standard_normal(40)
    k = np.arange(5)[:, None]
    n = np.arange(40)
    naive = np.sqrt(2 / 40) * (np.cos(np.pi * (2 * n + 1) * k / 80) * x).sum(axis=1)
    naive[0] /= np.sqrt(2)
    np.testing.assert_allclose(dct_matrix(40, 5) @ x, naive, atol=1e-12)


def test_hann_window_is_periodic_hann() -> None:
    w = hann_window(640)
    assert w[0] == 0.0 and w[320] == pytest.approx(1.0)
    np.testing.assert_allclose(w, np.hanning(641)[:-1], atol=1e-12)


def test_mfcc_shape_and_batch_consistency() -> None:
    x = np.random.default_rng(1).standard_normal((4, 16000)) * 0.1
    batch = mfcc(x)
    assert batch.shape == (4, 49, 10)
    for i in range(4):
        np.testing.assert_allclose(mfcc(x[i]), batch[i], atol=1e-12)


def test_mfcc_single_frame_matches_naive_computation() -> None:
    spec = load_spec()
    x = np.random.default_rng(2).standard_normal(16000) * 0.1
    t = 7
    seg = x[t * 320 : t * 320 + 640] * hann_window(640)
    padded = np.zeros(1024)
    padded[:640] = seg
    power = np.abs(np.fft.fft(padded)[:513]) ** 2
    naive = spec.dct @ np.log(spec.mel @ power + 1e-6)
    np.testing.assert_allclose(mfcc(x)[t], naive, atol=1e-10)


def test_mfcc_is_causal_per_frame() -> None:
    """Frame t depends only on samples [t*hop, t*hop + window): no centre padding."""
    x = np.random.default_rng(3).standard_normal(16000) * 0.1
    y = x.copy()
    y[8000:] += 1.0  # changes frames whose window reaches sample 8000
    a, b = mfcc(x), mfcc(y)
    first_changed = (8000 - 640) // 320 + 1
    np.testing.assert_array_equal(a[:first_changed], b[:first_changed])
    assert not np.allclose(a[first_changed], b[first_changed])


def test_torch_matches_numpy_reference() -> None:
    torch = pytest.importorskip("torch")
    from train.mfcc_torch import TorchMfcc

    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 16000)) * rng.uniform(0.001, 0.5, size=(8, 1))
    x[0] = 0.0  # digital silence: log(eps) floor
    ref = mfcc(x)
    out = TorchMfcc()(torch.from_numpy(x).float()).numpy()
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() < 1e-4
