"""Sample and feature ring buffers. LEARNING MODE: you write the bodies (see TODOs).

Concept
-------
A stream never ends, so we cannot keep "the whole signal". We only ever need a sliding
window: the last 640 samples (one MFCC window) and the last 49 frames (one model input).
A ring buffer keeps that window in a fixed, preallocated array. Instead of shifting the data
on every push (O(n) copy, and in Python an easy place to allocate), it keeps a write index
that wraps around: new data overwrites the oldest slot.

Reading "in time order" is the subtle part. Once the buffer has wrapped, the oldest element is
at the write index, not at index 0, so the logical sequence is arr[head:] followed by arr[:head].
Consumers (the FFT, the model) want oldest-to-newest, so `read_into` does that unwrapping with
at most two slice copies into a caller-owned array. Nothing is allocated in the hot loop.

Two buffers, two jobs:
  * SampleRingBuffer: raw samples, pushed 320 at a time (one hop), read as the latest 640.
  * FeatureRingBuffer: MFCC frames, pushed one at a time, read as the latest 49 x 10 block
    (float32, already normalised) for ORT. Normalising once per pushed frame (10 values) is
    far cheaper than normalising the 490 values at every inference.

The tests are in tests/test_ring_buffer.py. They fail until you implement these.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class SampleRingBuffer:
    """Fixed-capacity buffer of the most recent `capacity` samples."""

    def __init__(self, capacity: int, dtype: Any = np.float64) -> None:
        # TODO: preallocate `self._buf` (capacity,), a write index and a count of valid samples.
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.__init__")

    def __len__(self) -> int:
        """Number of valid samples held, at most `capacity`."""
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.__len__")

    @property
    def full(self) -> bool:
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.full")

    def reset(self) -> None:
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.reset")

    def push(self, x: NDArray[Any]) -> None:
        """Append `x` (1-D, len(x) <= capacity), overwriting the oldest samples.

        TODO: handle wrap-around with at most two slice assignments. Raise ValueError if
        len(x) > capacity. Do not allocate (no np.concatenate / np.roll).
        """
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.push")

    def read_into(self, out: NDArray[Any]) -> None:
        """Copy the held samples, oldest first, into `out` (shape (capacity,)).

        TODO: raise ValueError unless the buffer is full. Once full, the oldest sample is at
        the write index; copy buf[head:] then buf[:head].
        """
        raise NotImplementedError("TODO(learning mode): SampleRingBuffer.read_into")


class FeatureRingBuffer:
    """Fixed-capacity buffer of the most recent `n_frames` MFCC frames of width `n_mfcc`."""

    def __init__(
        self,
        n_frames: int,
        n_mfcc: int,
        mean: NDArray[np.float64] | None = None,
        std: NDArray[np.float64] | None = None,
    ) -> None:
        # TODO: preallocate a float32 (n_frames, n_mfcc) store. If mean/std are given, frames
        # are normalised as (frame - mean) / std at push time (spec 3.2). Both or neither.
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.__init__")

    def __len__(self) -> int:
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.__len__")

    @property
    def full(self) -> bool:
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.full")

    def reset(self) -> None:
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.reset")

    def push(self, frame: NDArray[np.float64]) -> None:
        """Append one (n_mfcc,) frame, normalising into the store without allocating.

        TODO: use np.subtract / np.divide with out= pointing at the destination row.
        """
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.push")

    def read_into(self, out: NDArray[np.float32]) -> None:
        """Copy the frames, oldest first, into `out` (shape (n_frames, n_mfcc), float32).

        TODO: raise ValueError unless full. Same two-slice unwrap as SampleRingBuffer.
        """
        raise NotImplementedError("TODO(learning mode): FeatureRingBuffer.read_into")
