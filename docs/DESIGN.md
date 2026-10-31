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

## M1 — Data, features, training, quantisation

| Decision | Alternatives | Why |
| --- | --- | --- |
| Held-out noise = `doing_the_dishes.wav` + `pink_noise.wav` (whole files) | Any 2 of the 6 files | One real and one synthetic recording, so the M3 streams contain both kinds of unseen noise. Never used in training, validation or `_silence_`. |
| The other 4 noise files are cut by time 80/10/10 into train/val/test segments | Draw `_silence_` and augmentation noise for every split from the same files | Val/test `_silence_` clips never share audio with training noise. The spec only requires the 2-file hold-out; this is an extra safeguard. |
| `_unknown_` and `_silence_` each sized to the mean keyword-class size of the split (about 1/12) | Fixed fractions of the whole dataset | Matches "about 1/12 of each split" and keeps classes balanced. `_unknown_` is a seeded random subsample. |
| Silence clip = random 1 s window of a noise segment times a gain in U(0, 1) | Full-volume noise only | Covers near-digital-silence up to full-level background noise. |
| Mel/DCT matrices written once to `features/*.npy`; NumPy, torch and (M2) streaming all load them | Recompute in each implementation | One definition; a torch-vs-NumPy test (< 1e-4) guards the rest. |
| HTK mel scale, peak-1 triangular filters, periodic Hann | Slaney mel, area-normalised filters | Simple and explicit. Normalisation stats absorb scale differences. |
| Normalisation mean/std from unaugmented training clips, stored in `features/config.json` | Per-clip normalisation | Streaming cannot see the whole clip; fixed stats work identically in both paths. |
| DS-CNN stem padding (5, 1) | No padding | Keeps a 25x5 feature map into the depthwise blocks. The spec gives kernel and stride only. Result: 23,756 parameters. |
| Cosine LR stepped per epoch, 30 epochs, batch 100, Adam 1e-3; best checkpoint by validation accuracy | Per-step schedule | Matches the spec. The test split is touched only by `train.eval_onnx`. |
| `torch.onnx.export(dynamo=False)`, opset 17 | New dynamo exporter | The legacy exporter gives the plain opset-17 graph ORT's quantiser handles; a test checks the opset. |
| QOperator, QInt8/QInt8, per-channel, MinMax, 200 train clips (as the spec says) | QDQ | The spec mandates QOperator. ORT warns that QInt8 activations and weights in QOperator "may lead to bad performance on x64". That is a latency question for M4; I will report it there rather than change the spec'd format silently. |
| No quantisation fallbacks tried | The spec's fallback list | INT8 validation accuracy was 93.32% vs FP32 93.20%, so the "loses more than 2 points" trigger never fired. The single attempt is logged in `results/quantization_attempts.json`. |
| Tuning and calibration decisions use validation only | | Honesty rule 3. |
