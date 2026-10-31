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
