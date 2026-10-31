from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tools.eval_stream import score
from tools.make_streams import (
    Event,
    build_stream,
    make_noise_bed,
    pick_utterances,
    read_events,
    speech_extent,
    write_events,
)

SR = 16000


def tone_clip(n_active: int = 8000, lead: int = 3000) -> NDArray[np.int16]:
    """1 s clip: quiet noise, then a 0.5 s loud tone starting at `lead`."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 5, SR)
    t = np.arange(n_active) / SR
    x[lead : lead + n_active] += 6000 * np.sin(2 * np.pi * 440 * t)
    return x.astype(np.int16)


def noise_files() -> dict[str, NDArray[np.float32]]:
    rng = np.random.default_rng(1)
    return {
        "a.wav": rng.normal(0, 0.1, 60 * SR).astype(np.float32),
        "b.wav": rng.normal(0, 0.3, 60 * SR).astype(np.float32),
    }


def test_speech_extent_finds_tone() -> None:
    lo, hi = speech_extent(tone_clip())
    assert abs(lo - 3000) <= 160 and abs(hi - 11000) <= 160


def test_noise_bed_has_target_rms_and_length() -> None:
    bed = make_noise_bed(noise_files(), 120 * SR, np.random.default_rng(3))
    assert len(bed) == 120 * SR
    assert 0.015 < np.sqrt(np.mean(bed**2)) < 0.025


def test_build_stream_layout_is_deterministic_and_respects_gaps() -> None:
    utts = [(tone_clip(), "yes" if i % 3 else "_unknown_", f"u{i}") for i in range(60)]
    s1, ev1, q1 = build_stream(utts, noise_files(), 120.0, np.random.default_rng(7))
    s2, ev2, q2 = build_stream(utts, noise_files(), 120.0, np.random.default_rng(7))
    assert np.array_equal(s1, s2) and ev1 == ev2 and q1 == q2
    assert len(s1) == 120 * SR and s1.dtype == np.int16
    assert len(ev1) > 10
    for a, b in zip(ev1, ev1[1:], strict=False):
        assert b.start_s - a.end_s >= 1.0 - 1e-9  # at least the 1 s minimum gap
    assert all(5.0 <= e.snr_db <= 20.0 for e in ev1)
    assert ev1[-1].end_s + 0.75 <= 120.0


def test_build_stream_snr_matches_request() -> None:
    utts = [(tone_clip(), "yes", f"u{i}") for i in range(40)]
    samples, events, _ = build_stream(utts, noise_files(), 90.0, np.random.default_rng(5))
    x = samples.astype(np.float64) / 32767
    e = events[3]
    speech = x[int(e.start_s * SR) : int(e.end_s * SR)]
    # The noise bed sits at RMS 0.02, so total RMS over the word is noise + speech in power.
    expected_total = 0.02 * np.sqrt(1 + 10 ** (e.snr_db / 10))
    assert np.sqrt(np.mean(speech**2)) == pytest.approx(expected_total, rel=0.1)


def test_build_stream_raises_when_utterances_run_out() -> None:
    with pytest.raises(ValueError):
        build_stream(
            [(tone_clip(), "yes", "u")] * 3, noise_files(), 600.0, np.random.default_rng(0)
        )


def test_events_csv_roundtrip(tmp_path: Path) -> None:
    ev = [Event(1.5, 2.25, "go", 12.3, "go/x.wav"), Event(5.0, 5.5, "_unknown_", 7.0, "cat/y.wav")]
    write_events(tmp_path / "t.csv", ev)
    assert read_events(tmp_path / "t.csv") == ev


def test_pick_utterances_has_no_repeats() -> None:
    kw = [f"yes/{i}.wav" for i in range(100)]
    unk = [f"cat/{i}.wav" for i in range(100)]
    picks = pick_utterances(kw, unk, 100, np.random.default_rng(0))
    assert len({p for p, _ in picks}) == 100
    assert {label for _, label in picks} == {"yes", "_unknown_"}


EVENTS = [
    Event(10.0, 10.6, "yes", 10.0, "a"),
    Event(20.0, 20.5, "no", 10.0, "b"),
    Event(30.0, 30.5, "_unknown_", 10.0, "c"),
    Event(40.0, 40.5, "go", 10.0, "d"),
]


def test_score_hits_misses_and_false_accept_categories() -> None:
    dets = [
        (10.9, "yes"),  # hit
        (11.3, "yes"),  # repeat inside the window (still <= end + 0.75): false accept
        (20.4, "up"),  # wrong label inside "no" window: false accept
        (30.6, "no"),  # keyword fired on an unknown word: false accept
        (50.0, "go"),  # nothing there: false accept
    ]  # "no" and "go" are missed
    m = score(EVENTS, dets, 3600.0)
    assert (m["keywords"], m["hits"], m["misses"]) == (3, 1, 2)
    assert m["miss_rate"] == pytest.approx(2 / 3)
    assert m["false_accepts"] == 4 and m["false_accepts_per_hour"] == pytest.approx(4.0)
    assert m["false_accept_categories"] == {
        "wrong_or_repeat_in_keyword_window": 2,
        "in_unknown_word_window": 1,
        "elsewhere": 1,
    }


def test_score_window_edges() -> None:
    m = score(EVENTS, [(9.75, "yes"), (21.25, "no")], 1800.0)  # start-0.25 and end+0.75
    assert m["hits"] == 2
    m = score(EVENTS, [(9.74, "yes"), (21.26, "no")], 1800.0)
    assert m["hits"] == 0 and m["false_accepts"] == 2
    assert m["false_accepts_per_hour"] == pytest.approx(4.0)
