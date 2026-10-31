"""Sample and feature ring buffers.

A stream never ends, so we only keep a sliding window: the last 640 samples (one MFCC window)
and the last 49 frames (one model input). A ring buffer keeps that window in a fixed,
preallocated array with a write index that wraps around: new data overwrites the oldest slot,
so nothing is shifted or allocated in the hot loop.

Reading in time order is the subtle part. Once the buffer has wrapped, the oldest element sits at
the write index, so the logical sequence is buf[head:] followed by buf[:head]. `read_into`
unwraps it with at most two slice copies into a caller-owned array.

  * SampleRingBuffer: raw samples, pushed one hop (320) at a time, read as the latest 640.
  * FeatureRingBuffer: MFCC frames, pushed one at a time, read as the latest 49 x 10 float32
    block, already normalised, for ORT. Normalising the 10 values of each pushed frame is far
    cheaper than normalising all 490 values at every inference.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


class SampleRingBuffer:
    """Fixed-capacity buffer of the most recent `capacity` samples."""

    def __init__(self, capacity: int, dtype: Any = np.float64) -> None:
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=dtype)
        self._head = 0  # next write position, which is also the oldest sample once full
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count == self._capacity

    def reset(self) -> None:
        self._head = 0
        self._count = 0

    def push(self, x: NDArray[Any]) -> None:
        n = len(x)
        if n > self._capacity:
            raise ValueError(f"push of {n} samples exceeds capacity {self._capacity}")
        first = min(n, self._capacity - self._head)
        self._buf[self._head : self._head + first] = x[:first]
        if first < n:  # wrapped: the rest goes to the start of the array
            self._buf[: n - first] = x[first:]
        self._head = (self._head + n) % self._capacity
        self._count = min(self._count + n, self._capacity)

    def read_into(self, out: NDArray[Any]) -> None:
        if not self.full:
            raise ValueError(f"buffer holds {self._count} of {self._capacity} samples")
        tail = self._capacity - self._head
        out[:tail] = self._buf[self._head :]
        out[tail:] = self._buf[: self._head]


class FeatureRingBuffer:
    """Fixed-capacity buffer of the most recent `n_frames` MFCC frames of width `n_mfcc`."""

    def __init__(
        self,
        n_frames: int,
        n_mfcc: int,
        mean: NDArray[np.float64] | None = None,
        std: NDArray[np.float64] | None = None,
    ) -> None:
        if (mean is None) != (std is None):
            raise ValueError("give both mean and std, or neither")
        self._n_frames = n_frames
        self._buf = np.zeros((n_frames, n_mfcc), dtype=np.float32)
        self._mean = mean
        self._std = std
        self._tmp = np.zeros(n_mfcc, dtype=np.float64)  # float64 scratch: round to float32 once
        self._head = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    @property
    def full(self) -> bool:
        return self._count == self._n_frames

    def reset(self) -> None:
        self._head = 0
        self._count = 0

    def push(self, frame: NDArray[np.float64]) -> None:
        row = self._buf[self._head]
        if self._mean is not None and self._std is not None:
            np.subtract(frame, self._mean, out=self._tmp)
            np.divide(self._tmp, self._std, out=row, casting="same_kind")
        else:
            row[:] = frame
        self._head = (self._head + 1) % self._n_frames
        self._count = min(self._count + 1, self._n_frames)

    def read_into(self, out: NDArray[np.float32]) -> None:
        if not self.full:
            raise ValueError(f"buffer holds {self._count} of {self._n_frames} frames")
        tail = self._n_frames - self._head
        out[:tail] = self._buf[self._head :]
        out[tail:] = self._buf[: self._head]
