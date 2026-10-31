import numpy as np
import pytest

from kws.ring_buffer import FeatureRingBuffer, SampleRingBuffer


def test_sample_buffer_fills_then_reads_in_order() -> None:
    buf = SampleRingBuffer(640)
    assert len(buf) == 0 and not buf.full
    buf.push(np.arange(320, dtype=np.float64))
    assert len(buf) == 320 and not buf.full
    buf.push(np.arange(320, 640, dtype=np.float64))
    assert len(buf) == 640 and buf.full
    out = np.empty(640)
    buf.read_into(out)
    np.testing.assert_array_equal(out, np.arange(640))


def test_sample_buffer_matches_tail_of_stream_after_wrapping() -> None:
    rng = np.random.default_rng(0)
    stream = rng.standard_normal(320 * 37)
    buf = SampleRingBuffer(640)
    out = np.empty(640)
    for i in range(37):
        buf.push(stream[320 * i : 320 * (i + 1)])
        if buf.full:
            buf.read_into(out)
            np.testing.assert_array_equal(out, stream[max(0, 320 * (i + 1) - 640) : 320 * (i + 1)])


def test_sample_buffer_handles_pushes_that_do_not_divide_capacity() -> None:
    rng = np.random.default_rng(1)
    stream = rng.standard_normal(1000)
    buf = SampleRingBuffer(64)
    out = np.empty(64)
    pos = 0
    for n in [50, 7, 64, 13, 30, 1, 64, 45, 20]:  # crosses the wrap point at odd offsets
        buf.push(stream[pos : pos + n])
        pos += n
        if buf.full:
            buf.read_into(out)
            np.testing.assert_array_equal(out, stream[pos - 64 : pos])


def test_sample_buffer_errors_and_reset() -> None:
    buf = SampleRingBuffer(8)
    with pytest.raises(ValueError):
        buf.push(np.zeros(9))
    buf.push(np.zeros(4))
    with pytest.raises(ValueError):
        buf.read_into(np.empty(8))  # not full yet
    buf.reset()
    assert len(buf) == 0 and not buf.full


def test_sample_buffer_keeps_dtype() -> None:
    buf = SampleRingBuffer(4, dtype=np.int16)
    buf.push(np.array([1, 2, 3, 4], dtype=np.int16))
    out = np.empty(4, dtype=np.int16)
    buf.read_into(out)
    np.testing.assert_array_equal(out, [1, 2, 3, 4])


def test_feature_buffer_order_after_wrapping() -> None:
    rng = np.random.default_rng(2)
    frames = rng.standard_normal((120, 10))
    buf = FeatureRingBuffer(49, 10)
    out = np.empty((49, 10), dtype=np.float32)
    for i, f in enumerate(frames):
        buf.push(f)
        assert len(buf) == min(i + 1, 49)
        if buf.full:
            buf.read_into(out)
            np.testing.assert_array_equal(out, frames[i - 48 : i + 1].astype(np.float32))


def test_feature_buffer_normalises_at_push() -> None:
    rng = np.random.default_rng(3)
    mean, std = rng.standard_normal(10), rng.uniform(0.5, 2.0, 10)
    frames = rng.standard_normal((60, 10)) * 3
    buf = FeatureRingBuffer(49, 10, mean, std)
    for f in frames:
        buf.push(f)
    out = np.empty((49, 10), dtype=np.float32)
    buf.read_into(out)
    np.testing.assert_allclose(out, (frames[-49:] - mean) / std, atol=1e-5)


def test_feature_buffer_errors_and_reset() -> None:
    buf = FeatureRingBuffer(49, 10)
    for _ in range(48):
        buf.push(np.zeros(10))
    with pytest.raises(ValueError):
        buf.read_into(np.empty((49, 10), dtype=np.float32))
    buf.push(np.zeros(10))
    assert buf.full
    buf.reset()
    assert len(buf) == 0 and not buf.full
    with pytest.raises(ValueError):
        FeatureRingBuffer(49, 10, np.zeros(10), None)  # mean without std
