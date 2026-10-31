"""M2 acceptance: streaming features and predictions vs the offline path.

Run: python -m tools.stream_parity     (test split; writes results/stream_feature_parity.json)

1. Raw MFCC parity: 50 test clips, streaming frames vs `mfcc_ref.mfcc`, max |difference|.
2. Model-input parity: every test clip pushed hop by hop through StreamingMFCC and
   FeatureRingBuffer, compared with the offline `eval_onnx.ort_features` tensor.
3. Prediction parity: FP32 and INT8 argmax from the streaming tensors vs the offline tensors
   (batch-1 ORT, 1 thread, as in `train.eval_onnx`). Predictions are not tuned on anything;
   this only checks that the two feature paths agree.
"""

import numpy as np
from numpy.typing import NDArray

from features import mfcc_ref
from kws.envinfo import REPO_ROOT, write_results
from kws.ring_buffer import FeatureRingBuffer
from kws.streaming_mfcc import StreamingMFCC
from train.data import load_dataset
from train.eval_onnx import FP32_PATH, INT8_PATH, make_session, ort_features, predict

N_PARITY_CLIPS = 50
SEED = 0


def stream_raw(s: StreamingMFCC, clip: NDArray[np.int16], hop: int) -> NDArray[np.float64]:
    frame = np.empty(10)
    frames = []
    s.reset()
    for i in range(0, len(clip), hop):
        if s.push(clip[i : i + hop], frame):
            frames.append(frame.copy())
    return np.stack(frames)


def stream_model_input(
    s: StreamingMFCC,
    buf: FeatureRingBuffer,
    clip: NDArray[np.int16],
    hop: int,
    out: NDArray[np.float32],
) -> None:
    frame = np.empty(10)
    s.reset()
    buf.reset()
    for i in range(0, len(clip), hop):
        if s.push(clip[i : i + hop], frame):
            buf.push(frame)
    buf.read_into(out[0, 0])


def main() -> None:
    spec = mfcc_ref.load_spec()
    cfg = spec.cfg
    hop = cfg["hop"]
    mean, std = mfcc_ref.norm_stats(spec)
    test = load_dataset().splits["test"]
    x, y = test.x, test.y

    s = StreamingMFCC(spec)
    buf = FeatureRingBuffer(cfg["n_frames"], cfg["n_mfcc"], mean, std)

    idx = np.random.default_rng(SEED).choice(len(x), N_PARITY_CLIPS, replace=False)
    diffs = [
        float(
            np.abs(
                stream_raw(s, x[i], hop) - mfcc_ref.mfcc(x[i].astype(np.float64) / 32768.0)
            ).max()
        )
        for i in idx
    ]

    offline = ort_features(x)
    streamed = np.empty_like(offline)
    for i in range(len(x)):
        stream_model_input(s, buf, x[i], hop, streamed[i : i + 1])
    input_diff = np.abs(streamed.astype(np.float64) - offline.astype(np.float64)).max(
        axis=(1, 2, 3)
    )

    models: dict[str, object] = {}
    for name, path in (("fp32", FP32_PATH), ("int8", INT8_PATH)):
        sess = make_session(path)
        p_off, p_str = predict(sess, offline), predict(sess, streamed)
        models[name] = {
            "predictions_differing": int((p_off != p_str).sum()),
            "offline_accuracy": float((p_off == y).mean()),
            "streaming_accuracy": float((p_str == y).mean()),
        }

    write_results(
        REPO_ROOT / "results" / "stream_feature_parity.json",
        {
            "split": "test",
            "raw_mfcc_parity": {
                "clips": N_PARITY_CLIPS,
                "clip_selection": f"numpy default_rng({SEED}).choice without replacement",
                "tolerance": 1e-6,
                "max_abs_diff": max(diffs),
                "mean_of_per_clip_max_abs_diff": float(np.mean(diffs)),
            },
            "model_input_parity": {
                "clips": int(len(x)),
                "max_abs_diff": float(input_diff.max()),
                "note": "normalised float32 tensor, streaming vs train.eval_onnx.ort_features",
            },
            "prediction_parity": {"clips": int(len(y)), "hop_samples": hop, **models},
        },
        ort_threads={"intra_op_num_threads": 1, "inter_op_num_threads": 1},
    )
    assert max(diffs) < 1e-6, "raw MFCC parity exceeded 1e-6"


if __name__ == "__main__":
    main()
