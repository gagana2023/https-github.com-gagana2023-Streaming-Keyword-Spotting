"""Torch MFCC used in training. Must match features/mfcc_ref.py to within 1e-4 (tested).

It loads the same mel.npy / dct.npy and window definition as the NumPy reference.
"""

import torch
from torch import nn

from features.mfcc_ref import FeatureSpec, load_spec, norm_stats


class TorchMfcc(nn.Module):
    """(B, n_samples) float waveform -> (B, n_frames, n_mfcc); optionally normalised."""

    window: torch.Tensor
    mel_t: torch.Tensor
    dct_t: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, spec: FeatureSpec | None = None, normalise: bool = False) -> None:
        super().__init__()
        spec = spec or load_spec()
        cfg = spec.cfg
        self.window_len: int = cfg["window"]
        self.hop: int = cfg["hop"]
        self.n_fft: int = cfg["n_fft"]
        self.eps: float = cfg["log_eps"]
        self.register_buffer("window", torch.from_numpy(spec.window).float())
        self.register_buffer("mel_t", torch.from_numpy(spec.mel.T.copy()).float())
        self.register_buffer("dct_t", torch.from_numpy(spec.dct.T.copy()).float())
        self.normalise = normalise
        if normalise:
            mean, std = norm_stats(spec)
            self.register_buffer("mean", torch.from_numpy(mean).float())
            self.register_buffer("std", torch.from_numpy(std).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = x.unfold(-1, self.window_len, self.hop) * self.window
        fft = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        power = fft.real**2 + fft.imag**2
        out = torch.log(power @ self.mel_t + self.eps) @ self.dct_t
        if self.normalise:
            out = (out - self.mean) / self.std
        return out
