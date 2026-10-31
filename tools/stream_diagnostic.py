"""Diagnostic: how well does the model classify words cut out of the validation stream?

Run: python -m tools.stream_diagnostic

Streaming misses and false accepts mix three effects: the model's accuracy on noisy audio, the
timing of the sliding window, and the smoothing/threshold. This isolates the first one. For
every word in the VALIDATION stream it takes the 1 s window centred on the word, classifies it
offline with the INT8 model (argmax, no threshold), and reports accuracy by SNR bucket, plus how
often an `_unknown_` word is labelled as a keyword. Compare with the clean-clip accuracy in
results/accuracy.json. The test stream is not used.
"""

import numpy as np

from kws.envinfo import REPO_ROOT, write_results
from kws.stream import CLASSES, SAMPLE_RATE, read_wav
from tools.make_streams import read_events, stream_paths
from train.eval_onnx import INT8_PATH, make_session, ort_features, predict

CLIP = SAMPLE_RATE
BUCKETS = ((5.0, 10.0), (10.0, 15.0), (15.0, 20.0001))


def main() -> None:
    wav_path, csv_path = stream_paths("val")
    stream = read_wav(wav_path)
    clips, labels, snrs = [], [], []
    for e in read_events(csv_path):
        start = int((e.start_s + e.end_s) / 2 * SAMPLE_RATE) - CLIP // 2
        if start < 0 or start + CLIP > len(stream):
            continue
        clips.append(stream[start : start + CLIP])
        labels.append(CLASSES.index(e.label))
        snrs.append(e.snr_db)
    y, snr = np.array(labels), np.array(snrs)
    pred = predict(make_session(INT8_PATH), ort_features(np.stack(clips)))
    keyword = y < 10

    buckets = {}
    for lo, hi in BUCKETS:
        m = keyword & (snr >= lo) & (snr < hi)
        buckets[f"{lo:g}-{min(hi, 20):g} dB"] = {
            "keywords": int(m.sum()),
            "accuracy": float((pred == y)[m].mean()),
        }
    write_results(
        REPO_ROOT / "results" / "stream_window_diagnostic.json",
        {
            "split": "val",
            "model": INT8_PATH.name,
            "window": "1 s centred on the word's midpoint, classified offline (argmax)",
            "keyword_accuracy": float((pred == y)[keyword].mean()),
            "keyword_accuracy_by_snr": buckets,
            "unknown_words": int((~keyword).sum()),
            "unknown_labelled_as_keyword": float((pred[~keyword] < 10).mean()),
            "unknown_accuracy": float((pred == y)[~keyword].mean()),
        },
        ort_threads={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    )


if __name__ == "__main__":
    main()
