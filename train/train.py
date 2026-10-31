"""Train the DS-CNN. Selects the checkpoint by validation accuracy; never touches the test split.

Run: python -m train.train [--epochs 30]
Writes models/dscnn.pt (best checkpoint) and results/train_summary.json.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from kws.envinfo import REPO_ROOT, write_results
from train.data import SEED, SplitData, augment, load_dataset
from train.mfcc_torch import TorchMfcc
from train.model import DSCNN, count_parameters

MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"


def to_features(frontend: TorchMfcc, pcm: np.ndarray[Any, Any]) -> torch.Tensor:
    """int16 or float PCM (B, 16000) -> normalised features (B, 1, 49, 10)."""
    x = torch.from_numpy(np.ascontiguousarray(pcm)).float()
    if pcm.dtype == np.int16:
        x = x / 32768.0
    return frontend(x).unsqueeze(1)


@torch.no_grad()
def evaluate(model: nn.Module, frontend: TorchMfcc, data: SplitData, batch: int = 500) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(data.y), batch):
        logits = model(to_features(frontend, data.x[i : i + batch]))
        correct += int((logits.argmax(1).numpy() == data.y[i : i + batch]).sum())
    return correct / len(data.y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-batches", type=int, default=None, help="smoke-test: cap per epoch")
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "dscnn.pt")
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "train_summary.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ds = load_dataset(seed=args.seed)
    train, val = ds.splits["train"], ds.splits["val"]

    frontend = TorchMfcc(normalise=True)
    model = DSCNN()
    n_params = count_parameters(model)
    print(f"DS-CNN trainable parameters: {n_params}")
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    n_batches = len(train.y) // args.batch_size
    if args.max_batches:
        n_batches = min(n_batches, args.max_batches)
    log: list[dict[str, float]] = []
    best_acc, best_epoch = -1.0, -1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(len(train.y))
        total_loss = 0.0
        for b in range(n_batches):
            idx = np.sort(order[b * args.batch_size : (b + 1) * args.batch_size])
            pcm = augment(train.x[idx].astype(np.float32) / 32768.0, ds.train_noise, rng)
            with torch.no_grad():
                feats = to_features(frontend, pcm)
            loss = loss_fn(model(feats), torch.from_numpy(train.y[idx]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        scheduler.step()
        val_acc = evaluate(model, frontend, val)
        seconds = time.perf_counter() - t0
        log.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / n_batches,
                "val_acc": val_acc,
                "lr": optimiser.param_groups[0]["lr"],
                "seconds": seconds,
            }
        )
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), args.out)
        print(
            f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}  val_acc {val_acc:.4f}"
            f"  {seconds:.1f}s{'  *' if best_epoch == epoch else ''}"
        )

    write_results(
        args.summary,
        {
            "parameter_count": n_params,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_schedule": "cosine decay to 0 over all epochs, stepped per epoch",
            "seed": args.seed,
            "train_clips": int(len(train.y)),
            "batches_per_epoch": n_batches,
            "best_epoch": best_epoch,
            "best_val_acc": best_acc,
            "mean_seconds_per_epoch": float(np.mean([e["seconds"] for e in log])),
            "per_epoch": log,
            "torch_threads": torch.get_num_threads(),
        },
    )


if __name__ == "__main__":
    main()
