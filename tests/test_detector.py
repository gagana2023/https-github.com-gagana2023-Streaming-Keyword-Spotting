import numpy as np

from kws.detector import NO_DETECTION, Detector, detect_from_logits


def logits_for(cls: int, strength: float = 10.0) -> np.ndarray:
    x = np.zeros(12, dtype=np.float32)
    x[cls] = strength
    return x


def test_fires_on_confident_keyword() -> None:
    d = Detector(threshold=0.8, smooth=1)
    assert d.update(logits_for(3), 0) == 3
    assert d.score > 0.99


def test_non_keyword_classes_never_fire() -> None:
    d = Detector(threshold=0.5, smooth=1)
    assert d.update(logits_for(10), 0) == NO_DETECTION  # _unknown_
    assert d.update(logits_for(11), 10_000) == NO_DETECTION  # _silence_


def test_below_threshold_does_not_fire() -> None:
    d = Detector(threshold=0.99, smooth=1)
    assert d.update(logits_for(2, 2.0), 0) == NO_DETECTION  # posterior about 0.4


def test_single_confident_inference_is_averaged_away() -> None:
    d = Detector(threshold=0.6, smooth=3)
    d.update(logits_for(10), 0)
    d.update(logits_for(10), 1600)
    assert d.update(logits_for(4), 3200) == NO_DETECTION  # mean of class 4 is about 1/3


def test_smoothing_fires_after_second_confident_inference() -> None:
    d = Detector(threshold=0.6, smooth=3)
    d.update(logits_for(10), 0)
    d.update(logits_for(4), 1600)
    assert d.update(logits_for(4), 3200) == 4


def test_suppression_window() -> None:
    d = Detector(threshold=0.8, smooth=1, suppress=16000)
    assert d.update(logits_for(0), 1000) == 0
    assert d.update(logits_for(0), 1000 + 15999) == NO_DETECTION
    assert d.update(logits_for(0), 1000 + 16000) == 0


def test_suppressed_inferences_still_update_history() -> None:
    d = Detector(threshold=0.8, smooth=2, suppress=16000)
    assert d.update(logits_for(0), 0) == 0
    d.update(logits_for(10), 1600)  # suppressed period, but history now has [yes, unknown]
    assert d.update(logits_for(10), 3200) == NO_DETECTION


def test_replay_matches_live_updates() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 4, size=(300, 12)).astype(np.float32)
    times = (np.arange(300, dtype=np.int64) + 1) * 1600
    live = Detector(0.7, 3)
    expected = []
    for i in range(300):
        c = live.update(logits[i], int(times[i]))
        if c != NO_DETECTION:
            expected.append((int(times[i]), c, live.score))
    assert detect_from_logits(logits, times, 0.7, 3) == expected
    assert len(expected) > 0
