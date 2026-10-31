"""Build long evaluation streams (WAV + ground-truth CSV) from one dataset split.

Run (needs the dataset; see `python -m train.data`):
    python -m tools.make_streams --split val  --minutes 20
    python -m tools.make_streams --split test --minutes 60

Layout of a stream:
  * Noise bed: random 5-15 s pieces of the two HELD-OUT noise files (never used in training,
    `_silence_` or augmentation), each scaled to the same RMS and cross-faded.
  * Utterances from the chosen split only (validation stream: validation utterances; test
    stream: test utterances). About 30 % are `_unknown_` words, the rest keywords.
  * Each utterance is scaled so its speech RMS sits at a random SNR of 5-20 dB over the bed,
    and placed so that the gap between the previous word's end and its start is 1-3 s.
  * With probability 0.1 an extra 5-10 s noise-only stretch is inserted after a word.

Ground truth: the speech extent of each clip (start/end) comes from a simple energy endpoint
detector on the clean clip, so it is approximate (see docs/DESIGN.md). Everything is seeded.
Outputs go to data/streams/ (gitignored); the recipe and checksums go to results/streams.json.
"""

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from kws.envinfo import REPO_ROOT, write_results

SR = 16000
HOP = 320
NOISE_RMS = 0.02  # RMS of the noise bed (full scale = 1.0)
SNR_RANGE_DB = (5.0, 20.0)
GAP_RANGE_S = (1.0, 3.0)
NOISE_ONLY_PROB = 0.1
NOISE_ONLY_RANGE_S = (5.0, 10.0)
NOISE_PIECE_RANGE_S = (5.0, 15.0)
UNKNOWN_FRACTION = 0.3
FADE = 1600  # 100 ms noise cross-fade
TAIL_S = 0.75  # room after the last word so its hit window fits inside the stream
SEEDS = {"val": 101, "test": 202}
STREAMS_DIR = REPO_ROOT / "data" / "streams"
RESULTS_PATH = REPO_ROOT / "results" / "streams.json"


@dataclass
class Event:
    start_s: float
    end_s: float
    label: str
    snr_db: float
    source: str


def speech_extent(
    clip: NDArray[np.int16], frame: int = 160, rel_thresh: float = 0.1
) -> tuple[int, int]:
    """(start, end) sample of the speech in a clip: 10 ms frames within 20 dB of the loudest."""
    x = clip.astype(np.float64) / 32768.0
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    if rms.max() == 0:
        return 0, len(clip)
    active = np.flatnonzero(rms >= rel_thresh * rms.max())
    return int(active[0]) * frame, (int(active[-1]) + 1) * frame


def make_noise_bed(
    noise: dict[str, NDArray[np.float32]], n_samples: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Random pieces of the RMS-normalised noise files, cross-faded, scaled to NOISE_RMS."""
    names = sorted(noise)
    unit = {
        k: v.astype(np.float64) / np.sqrt(np.mean(v.astype(np.float64) ** 2))
        for k, v in noise.items()
    }
    bed = np.zeros(n_samples)
    ramp = np.linspace(0.0, 1.0, FADE)
    pos = 0  # first sample of the bed not yet written
    while pos < n_samples:
        src = unit[names[rng.integers(len(names))]]
        length = int(rng.uniform(*NOISE_PIECE_RANGE_S) * SR)
        offset = int(rng.integers(0, len(src) - length))
        piece = src[offset : offset + length]
        if pos == 0:
            bed[: min(length, n_samples)] = piece[:n_samples]
            pos = min(length, n_samples)
            continue
        # The piece starts FADE samples early and fades in over what is already there.
        s = pos - FADE
        e = min(n_samples, s + length)
        piece = piece[: e - s]
        bed[s : s + FADE] = bed[s : s + FADE] * (1 - ramp) + piece[:FADE] * ramp
        bed[s + FADE : e] = piece[FADE:]
        pos = e
    return bed * NOISE_RMS


def build_stream(
    utterances: Sequence[tuple[NDArray[np.int16], str, str]],
    noise: dict[str, NDArray[np.float32]],
    duration_s: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int16], list[Event], list[tuple[float, float]]]:
    """Mix `utterances` [(clip, label, source)], used in order, over a noise bed.

    Returns (int16 samples, events in time order, noise-only stretches as (start_s, end_s)).
    Stops when the next word would not fit; raises if the utterances run out first.
    """
    n_total = int(duration_s * SR) // HOP * HOP
    mix = make_noise_bed(noise, n_total, rng)
    events: list[Event] = []
    quiet: list[tuple[float, float]] = []
    prev_end = 0
    for clip, label, source in utterances:
        gap = int(rng.uniform(*GAP_RANGE_S) * SR)
        snr = float(rng.uniform(*SNR_RANGE_DB))
        lo, hi = speech_extent(clip)
        start = prev_end + gap
        if start + (hi - lo) + int(TAIL_S * SR) > n_total:
            break
        x = clip.astype(np.float64) / 32768.0
        gain = NOISE_RMS * 10 ** (snr / 20) / np.sqrt(np.mean(x[lo:hi] ** 2))
        origin = start - lo  # stream position of clip sample 0 (may start before 0)
        a, b = max(origin, 0), min(origin + len(x), n_total)
        mix[a:b] += gain * x[a - origin : b - origin]
        prev_end = start + (hi - lo)
        events.append(Event(start / SR, prev_end / SR, label, snr, source))
        if rng.random() < NOISE_ONLY_PROB:
            extra = int(rng.uniform(*NOISE_ONLY_RANGE_S) * SR)
            quiet.append((prev_end / SR, (prev_end + extra) / SR))
            prev_end += extra
    else:
        raise ValueError(f"ran out of utterances after {len(events)} events")
    peak = float(np.abs(mix).max())
    if peak > 0.99:  # keep the stream inside int16; scales speech and noise alike
        mix *= 0.99 / peak
    return np.round(mix * 32767).astype(np.int16), events, quiet


def write_events(path: Path, events: Sequence[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "label", "snr_db", "source"])
        for e in events:
            w.writerow([f"{e.start_s:.3f}", f"{e.end_s:.3f}", e.label, f"{e.snr_db:.1f}", e.source])


def read_events(path: Path) -> list[Event]:
    with path.open(newline="") as f:
        return [
            Event(
                float(r["start_s"]), float(r["end_s"]), r["label"], float(r["snr_db"]), r["source"]
            )
            for r in csv.DictReader(f)
        ]


def stream_paths(split: str) -> tuple[Path, Path]:
    return STREAMS_DIR / f"{split}.wav", STREAMS_DIR / f"{split}_truth.csv"


def pick_utterances(
    keywords: list[str], unknown: list[str], n: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """n (relative path, label) pairs without repeats; each is unknown with prob 0.3."""
    kw = [keywords[i] for i in rng.permutation(len(keywords))]
    unk = [unknown[i] for i in rng.permutation(len(unknown))]
    out = []
    for _ in range(n):
        if rng.random() < UNKNOWN_FRACTION:
            rel = unk.pop()
            out.append((rel, "_unknown_"))
        else:
            rel = kw.pop()
            out.append((rel, rel.split("/", 1)[0]))
    return out


def main() -> None:
    from train import data

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--minutes", type=float, required=True)
    args = parser.parse_args()
    seed, duration_s = SEEDS[args.split], args.minutes * 60.0

    ds = data.load_dataset()
    root = data.dataset_root()
    val_set = data.read_list(root / "validation_list.txt")
    test_set = data.read_list(root / "testing_list.txt")
    pool = [
        r
        for r in data.list_utterances(root)
        if data.assign_split(r, val_set, test_set) == args.split
    ]
    keywords = [r for r in pool if data.label_of(r) != data.UNKNOWN]
    unknown = [r for r in pool if data.label_of(r) == data.UNKNOWN]

    rng = np.random.default_rng(seed)
    n_need = int(duration_s / 2.0)  # more than the ~3.4 s per word that actually fit
    picks = pick_utterances(keywords, unknown, n_need, rng)
    utterances = [(data.read_clip(root / rel), label, rel) for rel, label in picks]
    samples, events, quiet = build_stream(utterances, ds.held_out_noise, duration_s, rng)

    wav_path, csv_path = stream_paths(args.split)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, samples, SR, subtype="PCM_16")
    write_events(csv_path, events)

    labels = [e.label for e in events]
    entry = {
        "split": args.split,
        "seed": seed,
        "duration_s": len(samples) / SR,
        "events": len(events),
        "keywords": sum(label != "_unknown_" for label in labels),
        "unknown_words": labels.count("_unknown_"),
        "per_label": {k: labels.count(k) for k in sorted(set(labels))},
        "noise_only_stretches": len(quiet),
        "noise_only_seconds": sum(b - a for a, b in quiet),
        "held_out_noise_files": sorted(ds.held_out_noise),
        "recipe": {
            "noise_rms": NOISE_RMS,
            "snr_db_range": SNR_RANGE_DB,
            "gap_s_range": GAP_RANGE_S,
            "noise_only_prob": NOISE_ONLY_PROB,
            "noise_only_s_range": NOISE_ONLY_RANGE_S,
            "unknown_fraction": UNKNOWN_FRACTION,
        },
        "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
    }
    streams = {}
    if RESULTS_PATH.exists():
        streams = json.loads(RESULTS_PATH.read_text()).get("streams", {})
    streams[args.split] = entry
    write_results(RESULTS_PATH, {"streams": streams})
    print(
        f"{args.split}: {entry['duration_s'] / 60:.1f} min, {entry['keywords']} keywords, "
        f"{entry['unknown_words']} unknown words -> {wav_path}"
    )


if __name__ == "__main__":
    main()
