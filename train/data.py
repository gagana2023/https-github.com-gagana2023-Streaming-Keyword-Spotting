"""Speech Commands v2: 12-class splits, background-noise handling, caching, augmentation.

Run `python -m train.data` to download the dataset, build and cache the splits, compute the
feature normalisation statistics (written into features/config.json) and write
results/dataset.json.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from features import mfcc_ref
from kws.envinfo import REPO_ROOT, write_results

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
UNKNOWN = len(KEYWORDS)
SILENCE = UNKNOWN + 1
CLASSES = [*KEYWORDS, "_unknown_", "_silence_"]
SPLITS = ("train", "val", "test")

CLIP = 16000
SEED = 0
# Two of the six noise files are reserved for building evaluation streams (spec 3.1). One
# real recording and one synthetic noise, so streams contain both kinds of unseen noise.
HELD_OUT_NOISE = ("doing_the_dishes.wav", "pink_noise.wav")
# The remaining four files are cut by time into train/val/test segments, so val/test
# silence clips and augmentation noise never share audio with each other.
NOISE_SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
DATASET_VERSION = "speech_commands_v0.02"


class SplitData(NamedTuple):
    x: NDArray[np.int16]  # (N, CLIP) 16-bit PCM
    y: NDArray[np.int64]  # (N,) class index into CLASSES


def dataset_root(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "SpeechCommands" / DATASET_VERSION


def download(data_dir: Path = DATA_DIR) -> None:
    """Fetch and extract the dataset with torchaudio (no-op if already present)."""
    from torchaudio.datasets import SPEECHCOMMANDS

    data_dir.mkdir(parents=True, exist_ok=True)
    SPEECHCOMMANDS(str(data_dir), url=DATASET_VERSION, download=True)


# ---- split logic (pure functions, unit-tested) -------------------------------------------


def assign_split(rel_path: str, val_set: set[str], test_set: set[str]) -> str:
    """Official split: membership in validation_list.txt / testing_list.txt, else train."""
    if rel_path in test_set:
        return "test"
    if rel_path in val_set:
        return "val"
    return "train"


def label_of(rel_path: str) -> int:
    word = rel_path.split("/", 1)[0]
    return KEYWORDS.index(word) if word in KEYWORDS else UNKNOWN


def extra_class_size(n_keyword_clips: int) -> int:
    """Size of `_unknown_` and of `_silence_`: the mean keyword-class size."""
    return round(n_keyword_clips / len(KEYWORDS))


def select_clips(
    rel_paths: list[str], val_set: set[str], test_set: set[str], seed: int = SEED
) -> dict[str, list[tuple[str, int]]]:
    """Per split: every keyword clip plus a random subsample of unknown-word clips.

    The number of unknown clips (and, later, silence clips) is the mean keyword-class size,
    so `_unknown_` and `_silence_` are each about 1/12 of the split.
    """
    rng = np.random.default_rng(seed)
    by_split: dict[str, list[tuple[str, int]]] = {s: [] for s in SPLITS}
    unknown: dict[str, list[str]] = {s: [] for s in SPLITS}
    for rel in sorted(rel_paths):
        split = assign_split(rel, val_set, test_set)
        label = label_of(rel)
        if label == UNKNOWN:
            unknown[split].append(rel)
        else:
            by_split[split].append((rel, label))
    for split in SPLITS:
        n_extra = extra_class_size(len(by_split[split]))
        chosen = rng.choice(len(unknown[split]), size=n_extra, replace=False)
        by_split[split] += [(unknown[split][i], UNKNOWN) for i in sorted(chosen)]
    return by_split


def split_noise(
    noise: dict[str, NDArray[np.float32]],
) -> tuple[dict[str, dict[str, NDArray[np.float32]]], dict[str, NDArray[np.float32]]]:
    """Held-out files go to evaluation streams; the rest are cut by time into SPLITS.

    Returns ({split: {file: segment}}, {file: whole held-out recording}).
    """
    held_out = {k: v for k, v in noise.items() if k in HELD_OUT_NOISE}
    segments: dict[str, dict[str, NDArray[np.float32]]] = {s: {} for s in SPLITS}
    for name, wav in noise.items():
        if name in HELD_OUT_NOISE:
            continue
        a = int(len(wav) * NOISE_SPLIT_FRACTIONS["train"])
        b = a + int(len(wav) * NOISE_SPLIT_FRACTIONS["val"])
        segments["train"][name] = wav[:a]
        segments["val"][name] = wav[a:b]
        segments["test"][name] = wav[b:]
    return segments, held_out


def make_silence(
    segments: dict[str, NDArray[np.float32]], n: int, rng: np.random.Generator
) -> NDArray[np.int16]:
    """n one-second clips: random window of a random noise segment, random gain in [0, 1]."""
    names = sorted(segments)
    out = np.zeros((n, CLIP), dtype=np.int16)
    for i in range(n):
        seg = segments[names[rng.integers(len(names))]]
        start = rng.integers(0, len(seg) - CLIP + 1)
        gain = rng.uniform(0.0, 1.0)
        out[i] = np.round(seg[start : start + CLIP] * gain * 32767).astype(np.int16)
    return out


def augment(
    x: NDArray[np.float32],
    noise: list[NDArray[np.float32]],
    rng: np.random.Generator,
    max_shift: int = 1600,
    noise_prob: float = 0.8,
    max_noise_amp: float = 0.1,
) -> NDArray[np.float32]:
    """Training augmentation on a (B, CLIP) batch: +-100 ms shift, then background noise.

    With probability `noise_prob` a random window of a random training noise segment is
    scaled by an amplitude drawn from U(0, max_noise_amp) and added. Output is clipped to +-1.
    """
    b, n = x.shape
    out = np.zeros_like(x)
    shifts: list[int] = rng.integers(-max_shift, max_shift + 1, size=b).tolist()
    for i, s in enumerate(shifts):
        if s > 0:
            out[i, s:] = x[i, : n - s]
        elif s < 0:
            out[i, : n + s] = x[i, -s:]
        else:
            out[i] = x[i]
    for i in range(b):
        if rng.random() < noise_prob:
            seg = noise[rng.integers(len(noise))]
            start = rng.integers(0, len(seg) - n + 1)
            out[i] += seg[start : start + n] * rng.uniform(0.0, max_noise_amp)
    return np.clip(out, -1.0, 1.0)


# ---- IO ----------------------------------------------------------------------------------


def read_clip(path: Path) -> NDArray[np.int16]:
    """16-bit mono clip, zero-padded (or trimmed) to exactly one second."""
    data, sr = sf.read(path, dtype="int16")
    assert sr == 16000, f"{path}: expected 16 kHz, got {sr}"
    out = np.zeros(CLIP, dtype=np.int16)
    out[: min(CLIP, len(data))] = data[:CLIP]
    return out


def read_noise(root: Path) -> dict[str, NDArray[np.float32]]:
    files = sorted((root / "_background_noise_").glob("*.wav"))
    return {f.name: sf.read(f, dtype="float32")[0] for f in files}


def read_list(path: Path) -> set[str]:
    return set(path.read_text().split())


def list_utterances(root: Path) -> list[str]:
    return sorted(
        f"{d.name}/{f.name}"
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        for f in d.glob("*.wav")
    )


class Dataset(NamedTuple):
    splits: dict[str, SplitData]
    train_noise: list[NDArray[np.float32]]
    items: dict[str, list[tuple[str, int]]]  # utterance paths per split (no silence)
    held_out_noise: dict[str, NDArray[np.float32]]


def build_dataset(data_dir: Path = DATA_DIR, seed: int = SEED) -> Dataset:
    root = dataset_root(data_dir)
    val_set = read_list(root / "validation_list.txt")
    test_set = read_list(root / "testing_list.txt")
    items = select_clips(list_utterances(root), val_set, test_set, seed)
    segments, held_out = split_noise(read_noise(root))
    rng = np.random.default_rng(seed + 1)
    splits: dict[str, SplitData] = {}
    for split in SPLITS:
        clips = np.stack([read_clip(root / rel) for rel, _ in items[split]])
        labels = np.array([lab for _, lab in items[split]], dtype=np.int64)
        n_sil = extra_class_size(int((labels < UNKNOWN).sum()))
        sil = make_silence(segments[split], n_sil, rng)
        splits[split] = SplitData(
            np.concatenate([clips, sil]), np.concatenate([labels, np.full(n_sil, SILENCE)])
        )
    train_noise = [segments["train"][k] for k in sorted(segments["train"])]
    return Dataset(splits, train_noise, items, held_out)


def load_dataset(data_dir: Path = DATA_DIR, seed: int = SEED) -> Dataset:
    """build_dataset with an npz cache under data/cache (the WAV reading is the slow part)."""
    cache = data_dir / "cache" / f"seed{seed}.npz"
    items_path = cache.parent / f"seed{seed}_items.json"
    if not cache.exists():
        ds = build_dataset(data_dir, seed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, NDArray[np.generic]] = {}
        for s in SPLITS:
            arrays[f"x_{s}"], arrays[f"y_{s}"] = ds.splits[s]
        for i, seg in enumerate(ds.train_noise):
            arrays[f"noise_{i}"] = seg
        np.savez(cache, **arrays)  # type: ignore[arg-type]
        items_path.write_text(json.dumps(ds.items))
        return ds
    z = np.load(cache)
    splits = {s: SplitData(z[f"x_{s}"], z[f"y_{s}"]) for s in SPLITS}
    noise = [z[k] for k in sorted(k for k in z.files if k.startswith("noise_"))]
    raw_items = json.loads(items_path.read_text())
    items = {k: [(p, int(c)) for p, c in v] for k, v in raw_items.items()}
    all_noise = read_noise(dataset_root(data_dir))
    held_out = {k: v for k, v in all_noise.items() if k in HELD_OUT_NOISE}
    return Dataset(splits, noise, items, held_out)


# ---- statistics --------------------------------------------------------------------------


def compute_norm_stats(x_train: NDArray[np.int16]) -> tuple[list[float], list[float]]:
    """Per-coefficient mean/std of the MFCCs of the (unaugmented) training clips."""
    total = np.zeros(10)
    total_sq = np.zeros(10)
    count = 0
    for i in range(0, len(x_train), 1000):
        m = mfcc_ref.mfcc(x_train[i : i + 1000].astype(np.float64) / 32768.0).reshape(-1, 10)
        total += m.sum(axis=0)
        total_sq += (m**2).sum(axis=0)
        count += len(m)
    mean = total / count
    std = np.sqrt(total_sq / count - mean**2)
    return mean.tolist(), std.tolist()


def class_counts(y: NDArray[np.int64]) -> dict[str, int]:
    c = Counter(int(v) for v in y)
    return {name: c.get(i, 0) for i, name in enumerate(CLASSES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    download(args.data_dir)
    ds = load_dataset(args.data_dir, args.seed)

    mean, std = compute_norm_stats(ds.splits["train"].x)
    cfg_path = mfcc_ref.CONFIG_PATH
    cfg = json.loads(cfg_path.read_text())
    cfg["mean"], cfg["std"] = mean, std
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    mfcc_ref.load_spec.cache_clear()
    print(f"normalisation stats written to {cfg_path}")

    write_results(
        RESULTS_DIR / "dataset.json",
        {
            "dataset": DATASET_VERSION,
            "classes": CLASSES,
            "seed": args.seed,
            "counts": {s: class_counts(ds.splits[s].y) for s in SPLITS},
            "totals": {s: len(ds.splits[s].y) for s in SPLITS},
            "unknown_and_silence_size_rule": "mean keyword-class size of the split",
            "held_out_noise_files": list(HELD_OUT_NOISE),
            "noise_files_in_training_pool": sorted(
                f for f in read_noise(dataset_root(args.data_dir)) if f not in HELD_OUT_NOISE
            ),
            "noise_segment_fractions_by_time": NOISE_SPLIT_FRACTIONS,
            "norm_mean": mean,
            "norm_std": std,
        },
    )


if __name__ == "__main__":
    main()
