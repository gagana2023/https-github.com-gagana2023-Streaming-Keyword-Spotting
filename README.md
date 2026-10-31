# Streaming keyword spotting with INT8 inference

Work in progress: M0–M1 done (data, features, DS-CNN training, INT8 quantisation). The full
README (results table, limitations) is written at M4. See `docs/DESIGN.md` for decisions.

## Reproduce M1

```
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m train.run_all                                # about 10 min of training on a 20-thread laptop CPU
```

This downloads Speech Commands v2 (2.3 GB) into `data/`, trains for 30 epochs, writes
`models/dscnn_fp32.onnx` and `models/dscnn_int8.onnx` (gitignored) and the results in
`results/` (`dataset.json`, `train_summary.json`, `quantization.json`, `accuracy.json`,
`env.json`). Run the checks with `ruff check .`, `mypy` and `pytest`.
