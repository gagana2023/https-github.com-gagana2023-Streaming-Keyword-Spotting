"""Per-hop stage timing: preallocated recording and percentile summaries.

Times come from `time.perf_counter_ns` taken around each stage of every hop. Arrays are sized
up front so recording in the hot loop is one item assignment per stage and allocates nothing.
Warm-up (the first 50 inferences, and every hop before the 50th) is excluded from summaries.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

STAGES = ("features", "inference", "post")
DEADLINE_MS = 20.0
WARMUP_INFERENCES = 50


class StageTimes:
    def __init__(self, n_hops: int) -> None:
        self.ns = np.zeros((len(STAGES), n_hops), dtype=np.int64)
        self.ran_inference = np.zeros(n_hops, dtype=np.bool_)

    def summary(self, warmup_inferences: int = WARMUP_INFERENCES) -> dict[str, Any]:
        """p50/p95/p99/max in ms per stage and for the whole hop, after warm-up.

        `inference` and `post` are summarised over the hops that ran an inference; `features`
        and `total` over every hop. A deadline miss is a hop whose total exceeds 20 ms.
        """
        ran = np.flatnonzero(self.ran_inference)
        start = int(ran[warmup_inferences - 1]) + 1 if len(ran) >= warmup_inferences else None
        if start is None:
            return {"warmup_inferences": warmup_inferences, "hops_measured": 0}
        ms = self.ns[:, start:] / 1e6
        total = ms.sum(axis=0)
        has_inf = self.ran_inference[start:]
        out: dict[str, Any] = {
            "warmup_inferences": warmup_inferences,
            "hops_measured": int(total.size),
            "inferences_measured": int(has_inf.sum()),
            "features_ms": _percentiles(ms[0]),
            "inference_ms": _percentiles(ms[1][has_inf]),
            "post_ms": _percentiles(ms[2][has_inf]),
            "total_per_hop_ms": _percentiles(total),
            "deadline_ms": DEADLINE_MS,
            "deadline_misses": int((total > DEADLINE_MS).sum()),
        }
        return out


def _percentiles(x: NDArray[np.float64]) -> dict[str, float]:
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "max": float(np.max(x))}
