import numpy as np
import pytest
from numpy.typing import NDArray

from features import mfcc_ref
from kws.ring_buffer import FeatureRingBuffer
from kws.streaming_mfcc import StreamingMFCC

HOP = 320
N_CLIPS = 50


def synthetic_clips(n: int = N_CLIPS) -> NDArray[np.int16]:
    """Speech-like int16 clips: random tones plus noise at random levels, 1 s each."""
    rng = np.random.default_rng(42)
    t = np.arange(16000) / 16000.0
    clips = np.empty((n, 16000), dtype=np.int16)
    for i in range(n):
        f = rng.uniform(100, 3500, size=3)
        sig = sum(np.sin(2 * np.pi * fi * t + rng.uniform(0, 6.28)) for fi in f)
        sig = sig * rng.uniform(0.02, 0.2) + rng.standard_normal(16000) * rng.uniform(0.001, 0.05)
        clips[i] = np.clip(sig * 32768, -32768, 32767).astype(np.int16)
    return clips


def stream_clip(s: StreamingMFCC, clip: NDArray[np.int16]) -> NDArray[np.float64]:
    frame = np.empty(10)
    frames = []
    for i in range(0, len(clip), HOP):
        if s.push(clip[i : i + HOP], frame):
            frames.append(frame.copy())
    return np.stack(frames)


def test_first_hop_is_warmup_second_emits() -> None:
    s = StreamingMFCC()
    frame = np.full(10, np.nan)
    clip = synthetic_clips(1)[0]
    assert s.push(clip[:HOP], frame) is False
    assert np.isnan(frame).all()  # untouched during warm-up
    assert s.push(clip[HOP : 2 * HOP], frame) is True
    assert not np.isnan(frame).any()


def test_frame_count_matches_batch_framing() -> None:
    clip = synthetic_clips(1)[0]
    assert stream_clip(StreamingMFCC(), clip).shape == (49, 10)


def test_streaming_matches_batch_reference_on_50_clips() -> None:
    clips = synthetic_clips()
    ref = mfcc_ref.mfcc(clips.astype(np.float64) / 32768.0)
    s = StreamingMFCC()
    worst = 0.0
    for clip, r in zip(clips, ref, strict=True):
        s.reset()
        worst = max(worst, float(np.abs(stream_clip(s, clip) - r).max()))
    assert worst < 1e-6


def test_int16_and_float_inputs_agree() -> None:
    clip = synthetic_clips(1)[0]
    a = stream_clip(StreamingMFCC(), clip)
    s, frame, frames = StreamingMFCC(), np.empty(10), []
    f64 = clip.astype(np.float64) / 32768.0
    for i in range(0, 16000, HOP):
        if s.push(f64[i : i + HOP], frame):
            frames.append(frame.copy())
    np.testing.assert_allclose(np.stack(frames), a, atol=1e-9)


def test_reset_starts_a_fresh_stream() -> None:
    c1, c2 = synthetic_clips(2)
    s = StreamingMFCC()
    stream_clip(s, c1)
    s.reset()
    fresh = stream_clip(StreamingMFCC(), c2)
    np.testing.assert_array_equal(stream_clip(s, c2), fresh)


def test_wrong_hop_size_rejected() -> None:
    with pytest.raises(ValueError):
        StreamingMFCC().push(np.zeros(100, dtype=np.int16), np.empty(10))


def test_continuous_stream_windows_straddle_clip_boundaries() -> None:
    """Frames over a long stream equal the batch frames of the whole signal (no clip resets)."""
    clips = synthetic_clips(3)
    stream = np.concatenate(clips)
    ref = mfcc_ref.mfcc(stream.astype(np.float64) / 32768.0)
    got = stream_clip(StreamingMFCC(), stream)
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_feature_ring_buffer_holds_the_normalised_batch_features() -> None:
    clip = synthetic_clips(1)[0]
    spec = mfcc_ref.load_spec()
    mean, std = np.asarray(spec.cfg["mean"]), np.asarray(spec.cfg["std"])
    s = StreamingMFCC(spec)
    buf = FeatureRingBuffer(49, 10, mean, std)
    frame = np.empty(10)
    for i in range(0, 16000, HOP):
        if s.push(clip[i : i + HOP], frame):
            buf.push(frame)
    out = np.empty((49, 10), dtype=np.float32)
    buf.read_into(out)
    expected = mfcc_ref.normalize(mfcc_ref.mfcc(clip.astype(np.float64) / 32768.0), spec)
    np.testing.assert_allclose(out, expected.astype(np.float32), atol=1e-5)
