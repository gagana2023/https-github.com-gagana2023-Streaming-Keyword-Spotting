import numpy as np

from kws.stats import StageTimes


def test_summary_excludes_warmup_and_counts_deadline_misses() -> None:
    n = 200
    t = StageTimes(n)
    t.ns[0, :] = 1_000_000  # features: 1 ms every hop
    for h in range(4, n, 5):  # an inference every 5th hop
        t.ran_inference[h] = True
        t.ns[1, h] = 2_000_000
        t.ns[2, h] = 100_000
    t.ns[1, 4] = 50_000_000  # a slow first inference, inside the warm-up
    # Only 40 inferences exist, so with warmup=10 the first 10 (hops 4..49) are dropped.
    t.ns[1, 199] = 25_000_000  # deadline miss after warm-up
    s = t.summary(warmup_inferences=10)
    assert s["hops_measured"] == n - 50
    assert s["inferences_measured"] == 30
    assert s["inference_ms"]["max"] == 25.0
    assert s["inference_ms"]["p50"] == 2.0
    assert s["deadline_misses"] == 1
    assert np.isclose(s["features_ms"]["p50"], 1.0)


def test_summary_with_too_few_inferences() -> None:
    t = StageTimes(10)
    assert t.summary()["hops_measured"] == 0
