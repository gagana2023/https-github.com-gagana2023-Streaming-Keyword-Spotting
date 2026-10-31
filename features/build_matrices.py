"""Write the mel filterbank and DCT matrices to features/mel.npy and features/dct.npy.

Training and streaming both load these files, so there is a single definition of the
filterbank. Run: python -m features.build_matrices
"""

import json
from pathlib import Path

import numpy as np

FEATURES_DIR = Path(__file__).resolve().parent


def hz_to_mel(f: np.ndarray) -> np.ndarray:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular filters, shape (n_mels, n_fft // 2 + 1), peak 1, not area-normalised.

    Band edges are n_mels + 2 points equally spaced on the mel scale; each triangle rises
    linearly (in Hz) from one edge to the next and falls to the one after.
    """
    mel_lo, mel_hi = hz_to_mel(np.float64(fmin)), hz_to_mel(np.float64(fmax))
    edges = mel_to_hz(np.linspace(mel_lo, mel_hi, n_mels + 2))
    bin_hz = np.arange(n_fft // 2 + 1) * sr / n_fft
    fb = np.zeros((n_mels, bin_hz.size), dtype=np.float64)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bin_hz - lo) / (mid - lo)
        down = (hi - bin_hz) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Orthonormal DCT-II, shape (n_out, n_in): row k is the k-th basis vector."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    m = np.cos(np.pi * (2 * n + 1) * k / (2 * n_in))
    m *= np.sqrt(2.0 / n_in)
    m[0] *= 1.0 / np.sqrt(2.0)
    return m


def main() -> None:
    cfg = json.loads((FEATURES_DIR / "config.json").read_text())
    mel = mel_filterbank(
        cfg["sample_rate"], cfg["n_fft"], cfg["n_mels"], cfg["fmin"], cfg["fmax"]
    )
    dct = dct_matrix(cfg["n_mels"], cfg["n_mfcc"])
    np.save(FEATURES_DIR / "mel.npy", mel)
    np.save(FEATURES_DIR / "dct.npy", dct)
    print(f"mel {mel.shape} dct {dct.shape} written to {FEATURES_DIR}")


if __name__ == "__main__":
    main()
