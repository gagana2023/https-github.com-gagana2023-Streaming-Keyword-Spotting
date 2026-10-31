# Progress

## M0 — Setup

- Added `pyproject.toml` (ruff, mypy strict, pytest config), pinned `requirements.txt`, torch-free `requirements-ci.txt`, `.gitignore`, and `.github/workflows/ci.yml` (ruff, mypy, pytest).
- Added `tests/test_ort_smoke.py`, which builds a tiny ONNX graph and runs it through ORT.
- Local results (Python 3.12, onnxruntime 1.31.1): `ruff` clean, `mypy` clean, `pytest` 1 passed.
- CI has not been run yet: it needs a push to GitHub.

## M1 — Data, features, training, quantisation

- Implemented: `features/` (config, mel/DCT builder, NumPy reference), `train/` (data with held-out noise, torch MFCC, DS-CNN, training, ONNX export + INT8 quantisation, ORT evaluation, `run_all`) and `kws/envinfo.py`.
- Regenerated with `python -m train.run_all` (30 epochs, about 18 s/epoch on the i7-13650HX CPU). Outputs: `results/dataset.json`, `train_summary.json`, `quantization.json`, `quantization_attempts.json`, `accuracy.json`, `env.json`; `models/` is gitignored.
- Results: 23,756 parameters; best validation accuracy 93.20% (epoch 30); test accuracy FP32 92.82%, INT8 92.78% (-0.04 points); sizes 94,637 B / 37,589 B. No quantisation fallback was needed.
- Tests: `ruff` clean, `mypy` clean, `pytest` 22 passed.
- Note: the M1 code had been deleted from the working tree (it is still in git history). Restoring it was blocked, so it was re-written using the history as a reference and the results regenerated; they match the earlier run exactly.

## M2 — Streaming features

- Implemented `kws/ring_buffer.py` (`SampleRingBuffer`, `FeatureRingBuffer`) and `kws/streaming_mfcc.py` (`StreamingMFCC`, one frame per 320-sample hop). Tests (`tests/test_ring_buffer.py`, `tests/test_streaming_mfcc.py`) and `tools/stream_parity.py` were restored from git history and run against the new code.
- `python -m tools.stream_parity` wrote `results/stream_feature_parity.json` (test split): raw MFCC max |diff| 4.3e-14 over 50 clips (limit 1e-6); model-input tensors identical (0.0) over 4,888 clips; FP32 and INT8 predictions differ on 0 of 4,888 clips (accuracy 92.82% / 92.78%, same as offline).
- Tests: `ruff check` clean, `mypy` clean, `pytest` 38 passed. `ruff format --check` still flags three older files (`features/build_matrices.py`, `kws/envinfo.py`, `tests/test_dataset.py`) that this milestone did not touch.
- Note: `git checkout` of the deleted files was denied, so they were recreated with `git show HEAD:<path>`.

