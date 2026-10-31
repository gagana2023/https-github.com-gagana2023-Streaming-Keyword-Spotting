"""Regenerate everything in models/ and the M1 results with one command.

Run: python -m train.run_all [--epochs 30]

Steps: mel/DCT matrices -> dataset download, splits, normalisation stats -> training ->
ONNX export + INT8 quantisation -> ORT evaluation on the test split -> env.json.
"""

import argparse
import subprocess
import sys


def run(*module_and_args: str) -> None:
    cmd = [sys.executable, "-m", *module_and_args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run("features.build_matrices")
    run("train.data")
    run("train.train", "--epochs", str(args.epochs))
    run("train.export_quantize")
    run("train.eval_onnx")
    run("kws.envinfo")


if __name__ == "__main__":
    main()
