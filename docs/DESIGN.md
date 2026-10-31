# Design decisions

Each entry: the decision, the alternatives, and why.

## M0

### Python 3.12 in a local venv
Alternatives: system Python 3.14. Chosen 3.12 because torch/torchaudio/onnxruntime wheels
are reliably available for it; CI uses 3.12 too. The code still targets >=3.10.

### onnxruntime 1.31.1
Project 1 had no `requirements.txt` yet (only its spec), so this project sets the pin
(onnxruntime==1.31.1). Project 1 must use the same version.

### Two requirements files
`requirements.txt` has every pin including torch/torchaudio (CPU wheels). CI skips
training (spec Section 2), so `requirements-ci.txt` omits torch to keep CI fast. All other
versions are identical between the two files.

### mypy strict, ruff E/F/I/UP/B
Spec requires both to pass; strict mode catches typing slips early.

## M1

### Dataset handling: torchaudio downloads, soundfile reads
`torchaudio.datasets.SPEECHCOMMANDS` (as the spec requires) does the download and extraction.
I only use it for the download. I list the files and read them with soundfile, which is
already in the stack, so the loading path does not depend on torchaudio's audio backend
(whose requirements change between releases; I did not test its loader here). The official `validation_list.txt` / `testing_list.txt` define the split.

### Size of `_unknown_` and `_silence_`
Spec: "about 1/12 of each split". Chosen: each of the two extra classes has the mean
keyword-class size of that split (keyword clips / 10), so every one of the 12 classes is
about 1/12 of the split. Unknown clips are drawn uniformly at random from all other words
(fixed seed), per split, from that split's own files only.

### Held-out noise and where val/test silence comes from
Spec: 2 of the 6 background-noise files are held out for evaluation streams only. The spec
does not say where validation/test `_silence_` clips come from. Using the held-out files would
break "only for streams"; using the same audio as training silence would leak. Chosen: the
four non-held-out files are cut by time into 80% / 10% / 10% (train / val / test). Training
silence and augmentation noise come from the first segment, val/test silence from the others.
Held-out files: `doing_the_dishes.wav` (a real recording) and `pink_noise.wav` (synthetic),
fixed by name rather than drawn at random, so that streams contain one noise of each kind.
Alternative: random choice by seed. Rejected: it could hold out both synthetic files.
Silence clips get a random gain in [0, 1] so that near-silent clips exist.

### Features
- HTK mel scale, triangular filters with peak 1 (not area-normalised); the log and the
  per-coefficient normalisation absorb the scale.
- Periodic Hann window (`0.5 - 0.5 cos(2 pi n / N)`), the DFT-consistent variant.
- Matrices are stored as float64 `.npy` and loaded by training, evaluation and (M2) streaming.
  The torch twin casts them to float32; the tested difference to the NumPy reference is < 1e-4.
- Normalisation statistics are the mean/std over all frames of the unaugmented training clips,
  computed by `python -m train.data` and stored in `features/config.json`. They depend on
  the dataset seed, so the config is regenerated with the data.
- Evaluation (`eval_onnx`) uses the NumPy reference, not the torch features, because the
  streaming path in M2 is held to that reference.

### Model
Conv(10x4, stride 2x2) with padding (5, 1) gives a 25 x 5 map; four depthwise-separable
blocks keep 64 channels; global average pool; FC to 12. 23,756 trainable parameters
(BatchNorm running statistics are not parameters). The spec fixes everything except the
padding, which I chose so the stem does not discard the edges of the 49 x 10 input.

### Training
Augmentation runs on NumPy batches (shift, then noise at probability 0.8, amplitude
U(0, 0.1) against raw +-1 audio); features are computed by the torch twin. Adam 1e-3, cosine
decay over all epochs stepped per epoch, batch 100. The checkpoint with the best validation
accuracy is kept. The test split is not loaded by `train.train`.

### Export and quantisation
Legacy TorchScript exporter (`dynamo=False`) with `opset_version=17`, as the spec fixes. I did
not try the newer dynamo exporter. It prints a deprecation warning. Batch size is fixed at 1, matching the streaming use. Then ORT
`quant_pre_process`, then `quantize_static` (QOperator, QInt8/QInt8, per-channel, 200
random unaugmented training clips). Every run is appended to
`results/quantization_attempts.json` and judged on the validation split only.

### Reproducibility command
`python -m train.run_all` regenerates `models/` and all M1 results.

### CI and torch
CI has no torch. Modules that import it type-check there because `ignore_missing_imports` makes
torch an `Any`; `pyproject.toml` relaxes three mypy flags for `train.*` and `tests.*` so the
output is identical with and without torch. Torch-dependent tests use `importorskip`.
Checked locally with a torch-free venv built from `requirements-ci.txt`.

### Quantisation attempts (M1)
| # | Settings | Val FP32 | Val INT8 | Drop |
| --- | --- | --- | --- | --- |
| 1 | QOperator, QInt8/QInt8, per-channel, MinMax, 200 clips | 93.20% | 93.32% | -0.11 pts |

The first attempt is within the 2-point budget (INT8 is 0.11 points above FP32 on
validation, which is noise-level), so none of the fallbacks (more calibration data, per-tensor,
Entropy/Percentile) were tried. Test split, evaluated once after this choice: FP32 92.82%, INT8
92.78%, 43 of 4,888 predictions differ.

ORT logs `Please use QuantFormat.QDQ for activation type QInt8 and weight type QInt8. Or it
will lead to bad performance on x64.` The spec fixes QOperator with QInt8 activations (the
same recipe as Project 1), so I kept it. This is a likely reason INT8 could be slower than FP32
in M4; I have not measured it yet, so it stays a hypothesis until `bench.py` runs.

## M2 — Streaming features

Learning mode: the ring buffers and `StreamingMFCC` are written by the project owner. What I
(the assistant) provided is the stubs with concept notes, the tests, and the parity harness.

### Interfaces
- `SampleRingBuffer(capacity, dtype)`: `push(x)`, `read_into(out)` (oldest first, only when
  full), `full`, `__len__`, `reset`. Capacity 640 for the MFCC window.
- `FeatureRingBuffer(n_frames, n_mfcc, mean, std)`: float32 store; frames are **normalised at
  push time** (10 values per hop) instead of normalising 490 values at each inference.
- `StreamingMFCC.push(hop, out) -> bool`: one 320-sample hop in; the first hop is warm-up
  (False), every later hop writes one raw (unnormalised) 10-coefficient frame and returns True.
  Output is raw so it can be compared to `mfcc_ref.mfcc` at 1e-6 before normalisation.

### Decisions
- Raw frames out of `StreamingMFCC`, normalisation in `FeatureRingBuffer`: keeps the 1e-6
  parity test about the maths only, and keeps the normalisation in one place.
- Caller-owned `out` arrays and `read_into` everywhere, so the hot loop in M3/M4 can be
  allocation-free apart from what `np.fft.rfft` itself allocates (to be measured, not assumed).
- Tests use synthetic tones+noise clips (CI has no dataset or torch). The 50-real-clip parity
  and the full-test-set prediction check live in `tools/stream_parity.py`, which needs the
  dataset and writes `results/stream_feature_parity.json`.
- Prediction parity is checked for both FP32 and INT8 against `train.eval_onnx.ort_features`
  + the same batch-1 ORT sessions, so any disagreement can only come from the feature path.

### Reproduce
`python -m pytest tests/test_ring_buffer.py tests/test_streaming_mfcc.py` then
`python -m tools.stream_parity`.
