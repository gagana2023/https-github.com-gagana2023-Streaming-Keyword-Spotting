# Explain-back questions

Answer these without notes.

## M0 — Setup

1. Why does the spec pin exact versions, and why must onnxruntime match Project 1's version?
2. Why does CI install a requirements file without torch, and what does that mean for what CI can test?
3. What does `onnx.checker.check_model` verify that ORT loading alone might not?
4. In the smoke test, why compare to a NumPy computation instead of just checking the output shape?
5. Why is `ir_version` set explicitly on the test model, and what can go wrong if it is newer than the runtime supports?
