"""Per-hop stage timing: preallocated recording and percentile summaries.

Times come from `time.perf_counter_ns` taken around each stage of every hop. Arrays are sized
up front so recording in the hot loop is one item assignment per stage and allocates nothing.
Warm-up (the first 50 inferences, and every hop before the 50th) is excluded from summaries.
"""

import sys
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


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, in bytes.

    Linux `ru_maxrss` is in KiB and macOS's is in bytes; Windows has no `resource` module, so
    the peak working set is read from the Win32 API (also bytes).
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD,
        ]  # fmt: skip
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
