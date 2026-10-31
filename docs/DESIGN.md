# Design decisions

## M0 — Setup

| Decision | Alternatives | Why |
| --- | --- | --- |
| Exact pins in `requirements.txt` (torch 2.15.0, torchaudio 2.11.1, numpy 2.5.4, onnx 1.23.2, onnxruntime 1.31.1, ...) | Version ranges | Spec requires reproducible numbers; ORT 1.31.1 must match Project 1. |
| Separate torch-free `requirements-ci.txt` | Install torch in CI | CI skips training, and torch is a multi-GB install. Anything needing torch must be skipped or guarded when it is absent. |
| Dev venv on Python 3.12, CI on 3.12 | Python 3.14 (system default) | Safer wheel availability for onnx/torch; `requires-python >=3.10` still holds. |
| ORT smoke test builds the ONNX model in the test with `onnx.helper` | Commit a `.onnx` fixture | No binary in git, and the graph is readable (MatMul, Add, Relu with known weights). |
| `mypy --strict` over the whole repo, `ruff` rules E,F,I,UP,B | Looser checks | Spec requires both to pass; strict is cheaper to keep than to retrofit. |
| `models/` and `data/` gitignored | Commit models | Spec: scripts regenerate them. |
