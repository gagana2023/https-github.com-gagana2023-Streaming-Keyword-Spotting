# Progress

## M0 — Setup

- Added `pyproject.toml` (ruff, mypy strict, pytest config), pinned `requirements.txt`, torch-free `requirements-ci.txt`, `.gitignore`, and `.github/workflows/ci.yml` (ruff, mypy, pytest).
- Added `tests/test_ort_smoke.py`, which builds a tiny ONNX graph and runs it through ORT.
- Local results (Python 3.12, onnxruntime 1.31.1): `ruff` clean, `mypy` clean, `pytest` 1 passed.
- CI has not been run yet: it needs a push to GitHub.
