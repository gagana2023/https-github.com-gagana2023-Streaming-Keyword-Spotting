"""Posterior smoothing and keyword detection.

Each inference gives 12 logits. They go through a softmax into posteriors, which are averaged
over the last `smooth` inferences (a moving average: one noisy inference cannot fire on its
own, at the cost of up to `smooth` * infer-every hops of extra delay). A detection fires when
the top smoothed class is one of the 10 keywords and its smoothed posterior exceeds
`threshold`. After a detection, further detections are suppressed for `suppress` samples
(1 s by default) so that one spoken word, which stays in the 1 s window for several
inferences, is reported once. Posteriors keep being smoothed while suppressed.

Time is counted in samples, so results do not depend on wall-clock speed.
"""

import numpy as np
from numpy.typing import NDArray

N_CLASSES = 12
N_KEYWORDS = 10  # classes 0..9 are keywords; 10 = _unknown_, 11 = _silence_
NO_DETECTION = -1


class Detector:
    def __init__(
        self,
        threshold: float,
        smooth: int = 3,
        suppress: int = 16000,
        n_classes: int = N_CLASSES,
        n_keywords: int = N_KEYWORDS,
    ) -> None:
        if smooth < 1:
            raise ValueError("smooth must be >= 1")
        self.threshold = threshold
        self._smooth = smooth
        self._suppress = suppress
        self._n_keywords = n_keywords
        self._history = np.zeros((smooth, n_classes), dtype=np.float64)
        self._post = np.zeros(n_classes, dtype=np.float64)
        self._head = 0
        self._count = 0
        self._last_fire: int | None = None
        self.score = 0.0  # smoothed posterior of the class returned by the last update

    def reset(self) -> None:
        self._head = 0
        self._count = 0
        self._last_fire = None

    def update(self, logits: NDArray[np.float32], now: int) -> int:
        """Feed one inference's logits at stream time `now` (samples).

        Returns the detected class index, or NO_DETECTION. `score` holds its smoothed
        posterior (or that of the top class when nothing fired).
        """
        row = self._history[self._head]
        np.subtract(logits, np.max(logits), out=row)
        np.exp(row, out=row)
        np.divide(row, row.sum(), out=row)
        self._head = (self._head + 1) % self._smooth
        self._count = min(self._count + 1, self._smooth)
        # Until `smooth` inferences have arrived, average the ones we have.
        np.mean(self._history[: self._count], axis=0, out=self._post)

        top = int(np.argmax(self._post))
        self.score = float(self._post[top])
        if top >= self._n_keywords or self.score <= self.threshold:
            return NO_DETECTION
        if self._last_fire is not None and now - self._last_fire < self._suppress:
            return NO_DETECTION
        self._last_fire = now
        return top


def detect_from_logits(
    logits: NDArray[np.float32],
    times: NDArray[np.int64],
    threshold: float,
    smooth: int,
    suppress: int = 16000,
) -> list[tuple[int, int, float]]:
    """Replay recorded logits through a Detector: [(time_samples, class, score)].

    The threshold and smoothing window do not change the features or the model output, so a
    sweep records the logits once and replays them here instead of re-running ORT.
    """
    det = Detector(threshold, smooth, suppress)
    found: list[tuple[int, int, float]] = []
    for i in range(len(logits)):
        cls = det.update(logits[i], int(times[i]))
        if cls != NO_DETECTION:
            found.append((int(times[i]), cls, det.score))
    return found
