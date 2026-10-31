# Explain-back

## M0 — Setup

### What was built
- **Pinned environment.** Every dependency has an exact version so a result today can be reproduced later. A second, smaller list (`requirements-ci.txt`) leaves out torch because CI never trains.
- **CI.** On every push GitHub runs three gates: `ruff` (style and obvious bugs), `mypy` (type errors), `pytest` (behaviour).
- **ORT smoke test.** Builds a 3-node ONNX graph, `relu(x @ 2I - 1)`, and runs it with ONNX Runtime. Worked example: input `[0, 0.5, 1, 2]` becomes `[0, 1, 2, 4] - 1 = [-1, 0, 1, 3]`, and ReLU gives `[0, 0, 1, 3]`.

### Interview questions
1. **Why pin exact versions?** Quantised accuracy and latency depend on the ORT and numpy builds. Pinning lets anyone rerun a script and get comparable numbers.
2. **Why a separate CI requirements file?** CI only needs lint, types and non-training tests. Skipping torch saves minutes and gigabytes.
3. **Why build the test model in code?** There is no opaque binary in git. The expected output can be derived by hand.
4. **What does the smoke test prove?** The ORT install works, the CPU provider loads, and a model can go from graph to output.
5. **Why `mypy --strict`?** It is much cheaper to start strict than to annotate a finished codebase.
