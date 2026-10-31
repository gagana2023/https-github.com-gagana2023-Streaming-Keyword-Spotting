"""NumPy batch reference: a whole 1-second clip (or a batch of clips) -> 49 x 10 MFCCs.

Pipeline per frame: Hann window -> zero-padded FFT -> power spectrum -> mel filterbank ->
log(mel + eps) -> orthonormal DCT-II, first 10 coefficients. No centre padding, so a clip
of 16,000 samples gives 1 + (16000 - 640) // 320 = 49 frames. Everything is float64.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

FEATURES_DIR = Path(__file__).resolve().parent
CONFIG_PATH = FEATURES_DIR / "config.json"


def hann_window(n: int) -> NDArray[np.float64]:
    """Periodic Hann window (the DFT-friendly variant, as in scipy.signal.get_window)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


@dataclass(frozen=True)
class FeatureSpec:
    cfg: dict[str, Any]
    window: NDArray[np.float64]  # (window,)
    mel: NDArray[np.float64]  # (n_mels, n_fft // 2 + 1)
    dct: NDArray[np.float64]  # (n_mfcc, n_mels)


@lru_cache(maxsize=1)
def load_spec() -> FeatureSpec:
    cfg = json.loads(CONFIG_PATH.read_text())
    return FeatureSpec(
        cfg=cfg,
        window=hann_window(cfg["window"]),
        mel=np.load(FEATURES_DIR / "mel.npy"),
        dct=np.load(FEATURES_DIR / "dct.npy"),
    )


def frame_count(n_samples: int, window: int, hop: int) -> int:
    return 1 + (n_samples - window) // hop


def mfcc(x: NDArray[np.floating[Any]], spec: FeatureSpec | None = None) -> NDArray[np.float64]:
    """MFCCs of x with shape (..., n_samples) -> (..., n_frames, n_mfcc), unnormalised."""
    spec = spec or load_spec()
    cfg = spec.cfg
    frames = sliding_window_view(np.asarray(x, dtype=np.float64), cfg["window"], axis=-1)
    frames = frames[..., :: cfg["hop"], :] * spec.window
    fft = np.fft.rfft(frames, n=cfg["n_fft"], axis=-1)
    power = fft.real**2 + fft.imag**2
    logmel = np.log(power @ spec.mel.T + cfg["log_eps"])
    out: NDArray[np.float64] = logmel @ spec.dct.T
    return out


def norm_stats(spec: FeatureSpec | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    spec = spec or load_spec()
    if spec.cfg["mean"] is None or spec.cfg["std"] is None:
        raise RuntimeError("features/config.json has no mean/std yet; run `python -m train.data`")
    return np.asarray(spec.cfg["mean"]), np.asarray(spec.cfg["std"])


def normalize(m: NDArray[np.float64], spec: FeatureSpec | None = None) -> NDArray[np.float64]:
    """Per-coefficient (x - mean) / std using the training-set statistics in the config."""
    mean, std = norm_stats(spec)
    out: NDArray[np.float64] = (m - mean) / std
    return out
