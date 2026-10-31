"""Streaming MFCC: one frame per 320-sample hop.

The batch reference cuts a clip into 640-sample windows every 320 samples (50 % overlap), so
frame i covers samples [320*i, 320*i + 640). Streaming does the same incrementally: each new hop
completes exactly one window (the 320 samples already held plus the 320 just arrived), so a ring
buffer of the last 640 samples is enough and no frame is recomputed. The first hop cannot make a
frame (only 320 of 640 samples exist); the second hop makes frame 0, so the frame count matches
the batch reference exactly.

Per-frame maths is identical to features/mfcc_ref.py, one frame at a time:
Hann window -> rfft zero-padded to n_fft -> power -> mel filterbank -> log(mel + eps) -> DCT-II.
It uses the same mel/DCT matrices and window from `load_spec()`. int16 input is scaled by 1/32768
as in the batch path. Output is the raw, unnormalised MFCC; FeatureRingBuffer normalises.

Scratch arrays are preallocated and reused with `out=`. numpy's rfft allocates its result, which
is accepted here and measured in M4.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from features.mfcc_ref import FeatureSpec, load_spec
from kws.ring_buffer import SampleRingBuffer

INT16_SCALE = 1.0 / 32768.0


class StreamingMFCC:
    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self._spec = spec or load_spec()
        cfg = self._spec.cfg
        self._window = cfg["window"]
        self._hop = cfg["hop"]
        self._n_fft = cfg["n_fft"]
        self._eps = cfg["log_eps"]
        n_bins = self._n_fft // 2 + 1
        self._samples = SampleRingBuffer(self._window, dtype=np.float64)
        self._scaled = np.zeros(self._hop, dtype=np.float64)  # one hop as float64 in [-1, 1)
        self._frame = np.zeros(self._window, dtype=np.float64)  # latest window, oldest first
        self._power = np.zeros(n_bins, dtype=np.float64)
        self._imag_sq = np.zeros(n_bins, dtype=np.float64)
        self._mel = np.zeros(cfg["n_mels"], dtype=np.float64)

    def reset(self) -> None:
        """Forget all samples so the next hop is treated as the start of a new stream."""
        self._samples.reset()

    def push(self, hop: NDArray[Any], out: NDArray[np.float64]) -> bool:
        """Feed exactly one hop of samples (int16 or float; int16 is scaled by 1/32768).

        If this hop completes a window, write the (n_mfcc,) raw MFCC frame into `out` and
        return True; otherwise (warm-up) leave `out` untouched and return False.
        """
        if len(hop) != self._hop:
            raise ValueError(f"expected {self._hop} samples per hop, got {len(hop)}")
        if hop.dtype == np.int16:
            np.multiply(hop, INT16_SCALE, out=self._scaled)
        else:
            self._scaled[:] = hop
        self._samples.push(self._scaled)
        if not self._samples.full:
            return False

        self._samples.read_into(self._frame)
        np.multiply(self._frame, self._spec.window, out=self._frame)
        fft = np.fft.rfft(self._frame, n=self._n_fft)  # allocates; zero-pads to n_fft
        np.multiply(fft.real, fft.real, out=self._power)
        np.multiply(fft.imag, fft.imag, out=self._imag_sq)
        np.add(self._power, self._imag_sq, out=self._power)
        np.matmul(self._spec.mel, self._power, out=self._mel)
        np.add(self._mel, self._eps, out=self._mel)
        np.log(self._mel, out=self._mel)
        np.matmul(self._spec.dct, self._mel, out=out)
        return True
