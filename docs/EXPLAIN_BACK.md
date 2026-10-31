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

## M2 — Streaming features

### What was built
- **Sample ring buffer.** A fixed 640-slot array plus a write index `head`. A push of 320 samples writes at `head` (split into two slices if it crosses the end) and advances `head` modulo 640. Once full, the oldest sample is at `head`, so a time-ordered read is `buf[head:]` then `buf[:head]`. Worked example, capacity 4 holding 1,2,3,4: push `5,6` and the array is `[5,6,3,4]` with `head=2`; reading `buf[2:]` then `buf[:2]` gives `3,4,5,6`.
- **Streaming MFCC.** Frame i covers samples `[320i, 320i+640)`. Each hop supplies the newer half of a window and the ring buffer still holds the older half. Hop 1: only 320 samples, so no frame (warm-up). Hop 2: 640 samples, so frame 0. A 16,000-sample clip is 50 hops and 49 frames, the same count as the batch reference. The per-frame maths is the batch maths on one row, with the same `mel.npy`, `dct.npy` and Hann window.
- **Feature ring buffer.** The same wrap-around idea for 49 frames of 10 values, stored as float32. Each pushed frame is normalised once, `(x - mean) / std`, so an inference only copies 490 values in time order into the model input.
- **Parity (from `results/stream_feature_parity.json`).** Raw MFCC max difference 4.3e-14 over 50 clips; model-input tensors identical (difference 0.0) over 4,888 test clips; 0 differing FP32 or INT8 predictions.

### Interview questions
1. **Frame i covers which samples, and why 49 frames and a silent first hop?** Samples `[320i, 320i+640)`; with no padding, 1 + (16000-640)/320 = 49. The first hop gives only 320 of the 640 samples a window needs, so it is warm-up; every later hop completes one window.
2. **Why a ring buffer, and what does `read_into` do?** Appending and slicing the tail copies or allocates on every hop; a ring buffer overwrites the oldest slot in a preallocated array. After wrapping, the oldest element is at `head`, so `read_into` copies `buf[head:]` then `buf[:head]` into a caller-owned array.
3. **Why raw MFCC out, normalise in the feature buffer?** It keeps the parity test about the maths alone and normalises 10 values per hop instead of 490 per inference. It must use the training-set mean and std from `features/config.json`; without them the model sees inputs on the wrong scale and accuracy collapses.
4. **Parity is 4e-14, so why is the tolerance 1e-6 rather than 0?** The two paths do the same maths but in a different order (batched matmul and FFT versus per-frame), so float64 rounding differs at ~1e-14. The model input is float32 and rounded from float64 once in both paths, so those tensors came out identical and predictions matched exactly. Exact equality is a measured result here, not a guarantee.
5. **What does `stream_parity.py` prove and not prove?** It proves the streaming features and predictions equal the offline ones on 4,888 test clips. It does not prove real-time latency, allocation behaviour, behaviour on continuous noisy streams, or detection quality (M3/M4).


## M3 — Streaming loop and evaluation

### What was built
- **Streaming loop (`kws/stream.py`).** Per 320-sample hop: one MFCC frame, normalised into the feature ring buffer. Every 5th hop, once 49 frames exist, the buffer is copied in time order into the ORT input array, ORT runs, and the detector sees the 12 logits. The first inference is at hop 50 (16,000 samples), then every 100 ms. Everything is preallocated.
- **IOBinding.** Normally `session.run` builds a new output array on each call. Here the input and output NumPy arrays are registered with ORT once; each inference overwrites the input array in place and reads the output array.
- **Detector.** Softmax turns logits into 12 probabilities. The last `k` probability vectors are averaged. Worked example, `k = 3`, threshold 0.6, class "yes": posteriors for yes over three inferences of 0.1, 0.9, 0.95 average to 0.65 > 0.6, so it fires; with 0.1, 0.1, 0.95 the average is 0.38, so it does not. After a firing, nothing fires for 1 s (16,000 samples), because a spoken word stays inside the 1 s window for several inferences.
- **Evaluation streams.** Utterances are placed over a noise bed made from the two held-out noise files, each at a random 5-20 dB SNR, with 1-3 s gaps and some noise-only stretches. A ground-truth CSV lists each word's start, end and label.
- **Scoring.** A hit is the right label inside [start - 0.25 s, end + 0.75 s]. Everything else is a false accept. Worked example: word "yes" at 10.0-10.6 s; a "yes" detection at 10.9 s is a hit; a second "yes" at 11.3 s is a false accept; a "no" at 10.4 s is a false accept.
- **Threshold sweep.** The logits are recorded once, then replayed through the detector for 5 smoothing windows x 42 thresholds. The plot shows miss rate against false accepts/hour. Operating point: lowest miss rate with at most 10 false accepts/hour on the validation stream, which is k = 4, threshold 0.93.
- **Results (all from `results/`).** Validation: 38.6 % miss, 9 false accepts/hour (20 min). Test (1 hour, fixed point): INT8 46.7 % miss, 15 false accepts/hour; FP32 46.3 % and 13. These are far worse than clip accuracy; see the diagnostic in DESIGN.md.

### Interview questions
1. **What does the smoothing window trade off?** A longer window needs sustained evidence, so it rejects single noisy inferences (fewer false accepts), but it delays detection by up to `k` x 100 ms and can dilute a short word. In the validation sweep k = 1 was clearly worse at low false-accept rates; k = 3 to 5 were close, and k = 4 won under the chosen rule.
2. **Why suppress for 1 s after a detection?** One spoken word stays in the 1 s window for about ten consecutive inferences. Without suppression it would fire repeatedly, and every repeat would count as a false accept.
3. **How did you choose the threshold without touching the test set?** The sweep ran on a separate 20-minute stream built from validation utterances; a fixed rule picked the point; the test script only reads that point from a file. Caveat: on 20 minutes one false accept is 3/hour, so the choice is noisy.
4. **Why is streaming accuracy so much lower than the 92.8 % clip accuracy?** The streams use 5-20 dB SNR with noise the model never trained on, include many unknown words, and the model must decide continuously rather than on a clip centred on the word. On centred windows from the validation stream the same model gets 83.9 % on keywords, and 27 % of unknown words look like keywords. I did not retrain, so I report the result as measured.
5. **What could make the stream numbers optimistic or pessimistic?** Optimistic: the noise comes from two files reused over an hour, and word boundaries are heuristic. Pessimistic: 30 % unknown words is more than a natural mix, and the SNR range includes 5 dB. False accepts per hour on one hour of audio is a count of 15, so it carries large statistical uncertainty.
