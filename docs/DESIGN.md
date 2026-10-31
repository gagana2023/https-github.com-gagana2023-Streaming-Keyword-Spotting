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

## M2 — Streaming features

Implemented in `kws/ring_buffer.py` and `kws/streaming_mfcc.py`; tested by
`tests/test_ring_buffer.py` and `tests/test_streaming_mfcc.py`; checked on real data by
`tools/stream_parity.py`.

| Decision | Alternative | Why |
| --- | --- | --- |
| `StreamingMFCC` returns raw (unnormalised) frames; `FeatureRingBuffer` normalises at push time | Normalise inside the MFCC, or the 490 values at each inference | The 1e-6 parity test is about the maths only; 10 values per hop instead of 490 per inference; normalisation lives in one place. |
| Caller-owned `out` arrays, `read_into` | Return fresh arrays | Keeps the M3/M4 hot loop free of Python-side allocation. |
| Mean subtracted into a float64 scratch, divided straight into the float32 store | Subtract and divide in float32 | Each value is rounded to float32 once, as in offline `normalize(...).astype(float32)`, so model inputs are bit-identical (0.0 difference), not merely close. |
| Hop copied into a float64 scratch (int16 scaled by 1/32768) before the sample buffer | Buffer int16 and scale per window | One float64 path for int16 and float input; a test checks they agree. |
| Power as `re*re + im*im`, matmul with `out=` | `np.abs(fft)**2` | No temporaries. Only the `np.fft.rfft` result is allocated per frame; M4 measures that, it is not assumed away. |
| Unit tests use synthetic tone+noise clips; real-clip parity is a tool | Dataset in tests | CI has no dataset or torch. `python -m tools.stream_parity` needs the dataset and writes the results file. |
| Prediction parity for both FP32 and INT8, same batch-1 ORT sessions as `train.eval_onnx` | Compare features only | Any disagreement can then only come from the feature path. |

**Measured** (`results/stream_feature_parity.json`, test split): raw MFCC max |diff| 4.3e-14 over
50 clips (limit 1e-6); model-input max |diff| 0.0 over 4,888 clips; 0 of 4,888 FP32 and 0 of
4,888 INT8 predictions differ from the offline path.

Reproduce: `python -m pytest tests/test_ring_buffer.py tests/test_streaming_mfcc.py`, then
`python -m tools.stream_parity`.

## M3 — Streaming loop and evaluation

| Decision | Alternatives | Why |
| --- | --- | --- |
| Detector: softmax, moving average of the last `k` posteriors (fewer while fewer than `k` exist), fire if the top class is a keyword and its score > threshold, then 1 s suppression | Average logits; reset history after a detection | Posteriors are bounded in [0, 1], so one threshold means the same thing for any `k`. History keeps updating while suppressed, so the average stays current. |
| Time is counted in samples (`(hop index + 1) * 320`); a detection is stamped with the end of the 1 s window that fired | Wall-clock time | Results do not depend on machine speed. The 49 frames cover exactly the last 16,000 samples, so the stamp is the window end. |
| Inference is scheduled by hop count (`(h + 1) % infer_every == 0` and 49 frames available); first inference at hop 50 | Wall-clock timer | Deterministic and testable (`test_run_stream_schedule_and_replay`). |
| `ORTModel` binds the input and output arrays once and only writes into them | Rebind per call; `session.run` | No per-call allocation (spec 3.5). A test checks the bound path equals `session.run` after in-place writes. `threads=0` means all cores. |
| Sweep records logits once and replays them through the same `Detector` for every (k, threshold) | Re-run ORT per grid point | Neither parameter changes features or model output, so replay is exact (tested) and the whole 5 x 42 grid takes about 40 s. |
| Stream noise bed: random 5-15 s pieces of the 2 held-out files, each scaled to equal RMS, cross-faded 100 ms, level fixed at RMS 0.02 | Add noise per utterance | One continuous background, as in a real stream; noise-only stretches then contain real noise. The 1 h stream reuses about 155 s of noise audio many times, so the noise is unseen but not unique. |
| SNR is set per utterance by scaling the speech so its RMS over the speech extent is 5-20 dB (uniform) above the bed RMS | Fixed SNR | Spec 3.6. The clip's own recorded background is scaled with it; this is small but means the SNR is nominal. |
| Gap = 1-3 s between the previous word's end and the next word's start, measured on speech extents | Gap between clip boundaries | Matches "random 1-3 s gaps" literally. Hit windows of neighbouring words never overlap. 10 % of words are followed by an extra 5-10 s noise-only stretch. |
| Ground-truth start/end = energy endpoint detector (10 ms frames within 20 dB of the loudest frame) on the clean clip | Whole 1 s clip; forced alignment | The dataset has no word timings. The endpoints are approximate; the asymmetric hit window (-0.25 s, +0.75 s) absorbs some of that. |
| About 30 % of stream words are `_unknown_`, drawn from the full pool of that split's other-word clips (not only the 1/12 subsample) | Natural ratio (about 1/11) | Gives false accepts on unknown words a fair chance to show up. It makes the stream harder than a typical mix (see limitations). |
| Val stream (20 min) from validation utterances, test stream (60 min) from test utterances, seeds 101 and 202; no utterance repeats within a stream | One stream for both | Spec 3.6. Generation is deterministic: regenerating gave identical numbers (checksums in `results/streams.json`). |
| Scoring: each keyword matched once by its earliest same-label detection in [start - 0.25, end + 0.75]; every other detection is a false accept (categorised for diagnosis only) | Count repeats inside the window as part of the hit | Spec 3.6: "any other detection". |
| Operating-point rule, fixed before looking at the sweep: lowest miss rate with at most 10 false accepts/hour on the validation stream; ties go to fewer false accepts, then higher threshold, then smaller window. Grid: k in 1..5, 42 thresholds from 0.30 to 0.999 | Fixed k = 3; knee of the curve by eye | The 10/h cap is a number I chose, not one from the spec. On a 20-minute stream one false accept is 3/h, so the chosen point rests on 3 false accepts. The full grid is in `results/stream_eval_val.json`. |
| Test evaluation reads the threshold and k from `results/stream_eval_val.json`; FP32 is also scored at the same point | Re-tune per model | The test stream cannot influence the choice. INT8 is the primary (deployed) model. |

### Findings

- Chosen operating point: k = 4, threshold 0.93 (validation: 38.6 % miss, 9 false accepts/hour). On the test stream the same point gives 46.7 % miss (441 of 828 keywords hit) and 15 false accepts in 1 hour for INT8; FP32 gives 46.3 % and 13. INT8 and FP32 agree within noise.
- This is much worse than the 92.8 % clip accuracy suggests. `tools/stream_diagnostic.py` (validation stream only) classifies the 1 s window centred on each word: keyword accuracy is 83.9 % overall (73.3 % at 5-10 dB, 88.8 % at 10-15 dB, 90.5 % at 15-20 dB), and 27.4 % of unknown words are labelled as keywords. So the main cause is the model on noisier, unseen-noise audio, not the streaming code (M2 showed the streaming features and predictions match the offline path exactly). Smoothing and a high threshold add further misses on top.
- I did not retrain or change the model: that would change scope. One likely reason, not tested: training noise was added at amplitude 0-0.1 of the raw files, which is mostly a much higher SNR than the 5-20 dB streams, and pink noise (a held-out file) has no equivalent in the training noise pool.
- Not tested, so not a finding: whether training with lower-SNR augmentation would close the gap.
