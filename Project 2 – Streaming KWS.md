# Project 2 Spec: Streaming Keyword Spotting with INT8 Inference

Train a small DS-CNN keyword spotter, quantise it to INT8, and run it in a real-time Python streaming pipeline with measured latency and detection quality. Every number comes from a committed script. No C++.

## 0. Instructions for Claude Code

Read this whole spec before writing code. Then follow these rules for the entire project.

**Working rules**

1. Work one milestone at a time (Section 5). At the end of each milestone, stop, summarise what was built, show the test results, and wait for my go-ahead.
2. Clarity over speed. Do not add libraries beyond those listed in Section 2.
3. Make small commits with clear messages, one logical change each. Never squash history.
4. Keep `docs/DESIGN.md` up to date: each decision, the alternatives and why it was chosen.
5. Ask before changing scope, feature parameters, the dataset split or the metrics.
6. Python 3.10+ with type hints. Code must pass `ruff` and `mypy`.

**Honesty rules**

1. Never invent or estimate results. Every number in the README must come from a committed script that writes to `results/`.
2. Every results file records: date, git commit, CPU model, core count, OS, Python and package versions, ORT thread settings and the exact command.
3. Use the official test split only for final numbers. Tune thresholds and hyperparameters on validation data only.
4. Label all timings as measured on the stated host CPU in Python. Never present them as device or accelerator latency.
5. Report results that look bad, such as INT8 running slower than FP32, with an explanation. Do not hide them.

**Learning mode**

When I write "learning mode" for a task, write the tests and a stub with a clear TODO, explain the concept in 5–10 lines, and let me write the implementation. Use learning mode by default for: incremental framing and the per-frame MFCC (M2), the sample and feature ring buffers (M2), and posterior smoothing and detection logic (M3).

After each milestone, add 5 questions to `docs/EXPLAIN_BACK.md` that I should be able to answer without notes.

## 1. Goal, non-goals and resume claims

**Goal.** A 12-class keyword spotter for Google Speech Commands v2 that runs on a live-style audio stream in Python. It computes features incrementally with NumPy, runs INT8 inference through the ONNX Runtime Python API, smooths posteriors, and reports detections. Latency, memory and detection errors are measured under documented conditions.

**Non-goals:** C++, microcontroller deployment, microphone capture, beating published accuracy, a custom inference engine (that is Project 1), quantisation-aware training (stretch only).

**Resume claims this repo must support.** Final wording is adjusted to what the results actually show.

| Resume line | Must be proven by |
| --- | --- |
| Trained a \<X>K-param DS-CNN on Speech Commands: \<X>% FP32 and \<X>% INT8 accuracy after quantisation | `results/accuracy.json`, with the parameter count from the training script |
| Built a real-time streaming pipeline (incremental MFCC, ONNX Runtime) with \<P50> / \<P95> ms latency on \<CPU> | `results/latency_*.json` |
| Measured \<N> false accepts/hour on a \<X>-hour stream; INT8 ran \<X>× \<faster / slower> than FP32 | `results/stream_eval_test.json`, `results/latency_*.json` |

## 2. Stack

- **Python 3.10+:** torch and torchaudio (training, dataset download and loading), numpy (features, including `numpy.fft.rfft`), onnx, onnxruntime, soundfile, matplotlib, pytest, ruff, mypy.
- **Pinning:** pin exact versions in `requirements.txt` at M0. Use the same onnxruntime version as Project 1.
- **CI:** GitHub Actions runs `ruff`, `mypy` and `pytest`. CI skips training and uses a small committed test fixture.

## 3. Key design decisions

### 3.1 Dataset and labels

- **Source:** Speech Commands v2 via `torchaudio.datasets.SPEECHCOMMANDS`, using the official `validation_list.txt` and `testing_list.txt` splits.
- **12 classes:** yes, no, up, down, left, right, on, off, stop and go, plus `_unknown_` (all other words, subsampled to about 1/12 of each split) and `_silence_` (1-second clips cut from `_background_noise_`).
- **Held-out noise:** split the background noise files by file into a training set and a held-out set of 2 files. Use the held-out files only to build evaluation streams (Section 3.6), so stream noise is never seen in training.
- **Reproducibility:** fixed seeds, and per-class counts for each split recorded in `results/dataset.json`.

### 3.2 Features: one definition shared by training and streaming

All parameters live in `features/config.json`:

| Parameter | Value |
| --- | --- |
| Sample rate, clip length | 16 kHz, 1 s (16,000 samples) |
| Window | 640 samples (40 ms), Hann |
| Hop | 320 samples (20 ms) |
| FFT size | 1024 (zero-padded), power spectrum |
| Mel filterbank | 40 bands, 20–4000 Hz |
| Log | log(mel + 1e-6) |
| DCT | DCT-II (orthonormal), keep 10 coefficients |
| Output | 49 frames × 10 coefficients, no centre padding |
| Normalisation | per-coefficient mean and std from the training set, stored in the config |

- `features/build_matrices.py` writes the mel filterbank and DCT matrices to `features/mel.npy` and `features/dct.npy`. **Training and streaming both load these same files** instead of recomputing them.
- `features/mfcc_ref.py` is the NumPy batch reference, turning a whole 1-second clip into 49×10. The torch version used in training must match it to within 1e-4 (test).
- The streaming implementation computes one frame per hop and must match the batch reference to within 1e-6 on 50 clips (M2).

### 3.3 Model and training

- **DS-CNN:**
  1. Conv 10×4, stride 2×2, 64 channels, then BatchNorm and ReLU.
  2. 4 blocks, each a depthwise 3×3 conv, BatchNorm, ReLU, then a pointwise 1×1 conv with 64 channels, BatchNorm, ReLU.
  3. Global average pooling, then an FC layer to 12 classes.
- **Input** shape is 1×1×49×10. The script prints the exact parameter count, which goes on the resume.
- **Augmentation** (training only): random time shift of ±100 ms, and with probability 0.8 mix in training-set background noise at amplitude 0–0.1.
- **Optimisation:** Adam at lr 1e-3 with cosine decay, about 30 epochs, batch size 100. Select the checkpoint by validation accuracy and evaluate on the test split once.
- Training must run on a laptop CPU or a free Colab GPU. Log time per epoch.

### 3.4 Export and quantisation

- Use the same pipeline as Project 1: ONNX opset 17, then ORT `preprocess`, then `quantize_static` with QOperator format, QInt8 activations and weights, per-channel weights and 200 calibration clips from the training split.
- Evaluate FP32 and INT8 models with ORT Python on the test split.
- If INT8 loses more than 2 points, try in order: more calibration data, per-tensor versus per-channel, then Entropy or Percentile calibration. QAT is a stretch goal only. Log every attempt in `docs/DESIGN.md`.

### 3.5 Streaming pipeline (Python)

- **Input:** a 16 kHz mono 16-bit WAV file read with soundfile, fed in hops of 320 samples to simulate real time. A `--realtime` flag optionally sleeps between hops.
- **Sample ring buffer:** holds the last 640 samples. Each hop produces exactly one new MFCC frame.
- **Feature ring buffer:** 49×10. Every `infer_every` hops (default 5, which is 100 ms), copy the latest 49 frames in time order into a preallocated float32 input array and run ORT.
- **Detection:**
  - Smooth posteriors with a moving average over the last `k` inferences (default 3).
  - Fire a detection when the top smoothed posterior is a keyword class and exceeds `threshold`.
  - After each detection, suppress further detections for 1 s.
- **Memory:**
  - Preallocate all NumPy arrays at startup, and use `out=` arguments and in-place operations in the hot loop.
  - Create the ORT session once. Use ORT IOBinding (`session.io_binding()`, `bind_cpu_input`, `bind_output`) to reuse the input and output buffers.
  - Measure steady-state Python allocations with `tracemalloc`: bytes allocated per hop over 1,000 hops after warm-up. ORT's native allocations are not visible to tracemalloc and are excluded from the claim.
- **Baseline vs optimised:** the baseline recomputes all 49 frames at every inference, creates fresh arrays and calls plain `session.run`. The optimised version uses incremental frames, preallocated arrays and IOBinding. Also compare:
  - FP32 against INT8 models.
  - Intra-op threads 1 against all cores.
  - The garbage collector enabled against `gc.disable()` during the loop.
- **Timing:** `time.perf_counter_ns` around features, inference and post-processing for every hop. Exclude 50 warm-up inferences. Report p50, p95, p99 and max per stage and per hop.
- **Deadline:** a deadline miss is a hop whose total processing time exceeds 20 ms.
- **Memory and size reporting:** peak RSS from `resource.getrusage`, noting that the units differ between Linux and macOS, and the model file size.

### 3.6 Evaluation streams

- `tools/make_streams.py` builds long WAV files plus ground-truth CSVs with start, end and label for each utterance.
- **Validation stream** (about 20 min): built from validation-split utterances. Use it to choose the threshold and `k`.
- **Test stream** (1 hour): built from test-split utterances. Use it for final numbers only.
- **Stream content:** keywords and unknown words separated by random 1–3 s gaps, laid over held-out background noise at a random SNR of 5–20 dB, plus noise-only stretches. Fixed seeds.
- **Metrics:**
  - A hit is a detection with the correct label inside \[start − 0.25 s, end + 0.75 s\].
  - Miss rate is missed keywords divided by total keywords.
  - A false accept is any other detection, including a wrong label.
  - Report false accepts per hour.
- **Threshold choice:** sweep the threshold on the validation stream and plot false accepts per hour against miss rate. Pick the operating point there, then report test-stream numbers at that fixed threshold.

## 4. Repo layout and CLI

```
kws-stream/
  pyproject.toml  requirements.txt  README.md
  features/   config.json  build_matrices.py  mfcc_ref.py  mel.npy  dct.npy
  train/      data.py  model.py  train.py  export_quantize.py  eval_onnx.py
  kws/        ring_buffer.py  streaming_mfcc.py  ort_model.py  detector.py  stream.py  stats.py
  tools/      make_streams.py  eval_stream.py  sweep_threshold.py  bench.py
  tests/      pytest: features, ring buffers, detector, dataset split, stream generator, metrics
  models/     (gitignored; scripts regenerate them)
  results/    committed JSON/CSV/PNG + env info
  docs/       DESIGN.md  EXPLAIN_BACK.md  RESULTS.md
```

```
python -m kws.stream --model m.onnx --wav stream.wav --threshold 0.8 --smooth 3 --infer-every 5
                     --threads 1 --mode baseline|optimised [--no-gc] [--realtime]
                     --detections out.csv --timings out.json
```

## 5. Milestones

Each milestone ends with a stop and a summary. “Done when” is the acceptance test.

**M0 — Setup**

- `pyproject.toml`, pinned `requirements.txt` and GitHub Actions CI.
- **Done when:** CI is green and a test loads a tiny ONNX model through ORT.

**M1 — Data, features, training and quantisation** (Project 1 needs this model)

- Implement Sections 3.1–3.4: dataset split with held-out noise, matrices, NumPy reference and torch features, DS-CNN training, export and INT8 quantisation, and ORT evaluation.
- **Done when:** `results/dataset.json` and `results/accuracy.json` exist (FP32 accuracy, INT8 accuracy, parameter count, model sizes), and `models/` can be regenerated by one documented command.

**M2 — Streaming features** (learning mode for framing, the per-frame MFCC and the ring buffers)

- Sample and feature ring buffers, and a streaming MFCC that computes one frame per hop.
- **Done when:**
  - Streaming frames match the batch reference to within 1e-6 on 50 clips.
  - Running each 1-second test clip through the streaming path gives exactly the same predictions as offline ORT evaluation.
  - The parity numbers are written to `results/stream_feature_parity.json`.

**M3 — Streaming loop and evaluation** (learning mode for the detection logic)

- `kws.stream` in optimised mode, `make_streams.py`, `eval_stream.py` and `sweep_threshold.py`.
- **Done when:**
  - The threshold sweep plot on the validation stream is committed.
  - The chosen threshold is recorded.
  - `results/stream_eval_test.json` reports hits, misses, miss rate, false accepts and false accepts per hour on the 1-hour test stream.

**M4 — Profiling, optimisation and documentation**

- Add baseline mode. `bench.py` runs the matrix {baseline, optimised} × {FP32, INT8} × {1 thread, all cores} on the test stream, plus a GC on/off run, with 5 repetitions each.
- Run the tracemalloc steady-state check.
- Explain why INT8 is faster or slower than FP32 on this CPU.
- Write the README with: what the project is, how to reproduce, a results table generated from `results/`, limitations, and what did not help.
- **Done when:**
  - `results/latency_*.json` contains per-stage p50, p95, p99 and max, deadline misses, peak RSS, allocations per hop, model sizes and environment info.
  - A fresh clone reproduces everything except training using the README's commands.

## 6. Results, build order and self-check

**Build order:** run this project's M0–M1 early, because Project 1's M3 needs the quantised KWS model. The full interleaving is in the Project 1 tab, Section 6.

**Results files to commit**

| File | Contents |
| --- | --- |
| `results/env.json` | CPU, cores, OS, Python and package versions, git commit |
| `results/dataset.json` | Per-class counts per split, held-out noise files |
| `results/accuracy.json` | FP32 and INT8 test accuracy, parameter count, model sizes, calibration settings |
| `results/stream_feature_parity.json` | Streaming vs batch feature difference and prediction agreement |
| `results/stream_eval_val.json` + `threshold_sweep.png` | Validation sweep and chosen operating point |
| `results/stream_eval_test.json` | Hits, misses, miss rate, false accepts, false accepts/hour, stream length |
| `results/latency_<mode>_<model>_<threads>.json` | Per-stage p50, p95, p99 and max, deadline misses, peak RSS, allocations per hop |

**Fill in the resume from these files only.** “\<X>× faster/slower” is the ratio of INT8 to FP32 p50 inference time, in optimised mode, with 1 thread.

**Interview self-check.** I should be able to answer these without notes:

- Walk from raw samples to one MFCC frame, and explain why each step exists.
- Why does incremental feature computation help, and by how much on my machine?
- How did I make sure training and streaming compute the same features?
- What do the smoothing window and suppression period trade off, and how did I pick them without touching the test set?
- Why was INT8 faster or slower than FP32 here, and what would change on an MCU or an NPU?
- Which stage dominates latency at p95, and how much does Python's garbage collector affect p95 and p99?
- How was the evaluation stream built, and what could make its numbers optimistic?
- What did ORT give me, and what did I write myself?
