"""Streaming MFCC: one frame per 320-sample hop. LEARNING MODE: you write the bodies.

Concept
-------
Framing: the batch reference cuts a clip into windows of 640 samples every 320 samples
(50 % overlap), so frame i covers samples [320*i, 320*i + 640). A 16,000-sample clip gives
1 + (16000 - 640) // 320 = 49 frames, with no padding at the edges.

Streaming does the same thing incrementally. Samples arrive 320 at a time. Each new hop
completes exactly one new window: the 320 samples we already had plus the 320 just arrived.
So a ring buffer of the last 640 samples is enough, and no frame is ever recomputed. The first
hop cannot make a frame yet (only 320 of 640 samples exist); the second hop makes frame 0.
That is why the output frame count matches the batch reference exactly.

Per-frame MFCC (identical maths to features/mfcc_ref.py, one frame at a time):
  window (Hann, 640) -> rfft zero-padded to 1024 -> power |X|^2 -> mel filterbank (40) ->
  log(mel + 1e-6) -> DCT-II, first 10 coefficients.
Use the SAME matrices from `load_spec()` (mel.npy, dct.npy) and the same Hann window; do not
recompute them. int16 input is scaled by 1/32768 exactly as the batch path does.

Output is the raw, unnormalised MFCC (so it can be compared to `mfcc_ref.mfcc` to 1e-6).
Normalisation happens in FeatureRingBuffer.push.

The hot loop must not allocate: preallocate the 640-sample window, the FFT input and the
intermediate vectors in __init__ and use `out=` arguments (np.multiply, np.matmul, np.log).
numpy's rfft allocates its result; that is accepted here and measured in M4.

Tests are in tests/test_streaming_mfcc.py. They fail until you implement this.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from features.mfcc_ref import FeatureSpec


class StreamingMFCC:
    def __init__(self, spec: FeatureSpec | None = None) -> None:
        # TODO: load_spec() if spec is None; build a SampleRingBuffer(window) and preallocate
        # every scratch array the per-frame computation needs.
        raise NotImplementedError("TODO(learning mode): StreamingMFCC.__init__")

    def reset(self) -> None:
        """Forget all samples so the next hop is treated as the start of a new stream."""
        raise NotImplementedError("TODO(learning mode): StreamingMFCC.reset")

    def push(self, hop: NDArray[Any], out: NDArray[np.float64]) -> bool:
        """Feed exactly `hop` samples (int16 or float; int16 is scaled by 1/32768).

        If this hop completes a window, write the (n_mfcc,) raw MFCC frame into `out` and
        return True; otherwise (warm-up) leave `out` untouched and return False.
        Raise ValueError if len(hop) != the configured hop size.
        """
        raise NotImplementedError("TODO(learning mode): StreamingMFCC.push")
