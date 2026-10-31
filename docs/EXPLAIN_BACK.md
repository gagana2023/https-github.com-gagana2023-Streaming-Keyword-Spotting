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

## M1 — Data, features, training, quantisation

### What was built
- **Dataset split.** Speech Commands v2 ships `validation_list.txt` and `testing_list.txt`; everything else is train, and speakers do not cross splits. Keywords are kept whole, other words become `_unknown_` (a random subsample of about 1/12), and `_silence_` is cut from background-noise recordings. Two noise files are never used for training: they are reserved for the M3 streams. Worked example: if a split has 3,000 keyword clips over 10 classes, `_unknown_` and `_silence_` get 300 each.
- **MFCC, one shared definition.** A 1 s clip is 16,000 samples. A 640-sample window slides by 320, giving 1 + (16000 - 640) / 320 = 49 frames. Per frame: multiply by a Hann window (reduces edge leakage), zero-pad to 1024 and take the FFT, square to get power, sum into 40 mel bands (resolution like human hearing), take the log (loudness is perceived on a log scale), then a DCT keeps 10 coefficients (decorrelates the bands and compacts the spectral envelope). Output is 49x10. Tiny example: a frame whose 40 log-mel values are all equal puts all its DCT energy in coefficient 0 and about 0 in the other nine.
- **Why the matrices are files.** `mel.npy` and `dct.npy` are written once. The NumPy reference, the torch training code and (next milestone) the streaming code load the same files, so they cannot drift. A test requires torch to match NumPy within 1e-4.
- **DS-CNN.** A normal conv (10x4, stride 2), then 4 depthwise-separable blocks: a 3x3 conv per channel plus a 1x1 conv that mixes channels. That costs about 9C + C² weights instead of 9C², which is why the whole net is 23,756 parameters.
- **Augmentation.** Shift up to ±100 ms, and with probability 0.8 add training-set noise at amplitude 0 to 0.1, so the model tolerates timing offsets and background noise.
- **Quantisation.** ONNX export, ORT pre-processing, then static INT8: 200 training clips are run through the model to record each activation's range, and weights and activations are stored as int8 with scales (per channel for weights). Example: a range of [-2, 2] mapped to [-128, 127] has scale about 0.0314, so 0.5 becomes round(0.5 / 0.0314) = 16.
- **Results (all from `results/`).** 23,756 parameters. Test accuracy FP32 92.82%, INT8 92.78% (43 of 4,888 predictions differ). Model sizes 94,637 B (FP32) and 37,589 B (INT8).

### Interview questions
1. **Why a fixed mean/std from the training set instead of normalising each clip?** Streaming never has the whole clip, and training and inference inputs must be scaled identically. Fixed stats work in both paths.
2. **Why hold out noise files instead of just using separate clips?** The M3 streams add noise; if the model had trained on that exact noise, false-accept numbers would be optimistic. Whole unseen files make the noise genuinely new.
3. **Why does depthwise-separable convolution suit a small KWS model?** It cuts parameters and multiply-adds by roughly an order of magnitude versus a full 3x3 conv at 64 channels, with little accuracy loss.
4. **What does static quantisation need calibration data for?** To choose the scale of each activation tensor, since unlike weights their ranges are unknown until data flows through. Unrepresentative calibration data clips or wastes range.
5. **How do you know the INT8 model is not worse?** The same test clips go through ORT at batch 1 for both models: 92.82% vs 92.78%, with 43 differing predictions. Tuning used validation only, and the test split was evaluated once.
