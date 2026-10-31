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
