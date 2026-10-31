# Explain-back questions

Answer these without notes.

## M0 — Setup

1. Why does the spec pin exact versions, and why must onnxruntime match Project 1's version?
2. Why does CI install a requirements file without torch, and what does that mean for what CI can test?
3. What does `onnx.checker.check_model` verify that ORT loading alone might not?
4. In the smoke test, why compare to a NumPy computation instead of just checking the output shape?
5. Why is `ir_version` set explicitly on the test model, and what can go wrong if it is newer than the runtime supports?

## M1 — Data, features, training and quantisation

1. Why are the mel and DCT matrices saved as `.npy` files that training and streaming both load, instead of each side computing them? What bugs does this rule out, and which bugs does it not rule out?
2. Why are 2 of the 6 noise files held out, and why did I also cut the other 4 by time into train/val/test segments? What would be optimistic about the numbers if val/test silence came from the same audio as training noise?
3. Where do the normalisation mean and std come from, why must they come from the training set only, and what breaks in M2 if the streaming path forgets to apply them?
4. In `quantize_static` with QOperator, QInt8 and per-channel weights: what does calibration actually measure, what does per-channel change compared with per-tensor, and why do I choose the checkpoint, thresholds and quantisation settings on validation rather than test?
5. The torch features match the NumPy reference only to 1e-4, not exactly. Why is that acceptable for training but why does eval_onnx use the NumPy reference?

## M2 — Streaming features

1. Frame i of a clip covers which samples, and why does a 16,000-sample clip give exactly 49 frames? Why does the streaming path emit nothing on the first hop and one frame on every hop after that?
2. Why is a ring buffer used instead of appending to an array and slicing the tail? Once it has wrapped, where is the oldest element, and what does `read_into` do to return samples in time order?
3. Why does the streaming MFCC return raw coefficients and leave normalisation to the feature ring buffer? Why is normalising at push time cheaper than at inference time, and which M1 numbers must it use?
4. The streaming frames match the batch reference to far better than 1e-6, yet the tolerance is 1e-6 and not exactly 0. What is different between the two code paths, and why is exact equality still achievable for the model-input tensor and the predictions?
5. What does `tools/stream_parity.py` prove, and what does it not prove? (Think: real-time timing, allocations, noise/streams, and whether the model is any good.)
