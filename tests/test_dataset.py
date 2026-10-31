import numpy as np

from train.data import (
    CLIP,
    HELD_OUT_NOISE,
    KEYWORDS,
    SPLITS,
    UNKNOWN,
    assign_split,
    augment,
    extra_class_size,
    label_of,
    make_silence,
    select_clips,
    split_noise,
)


def _fake_utterances() -> list[str]:
    words = [*KEYWORDS, "bed", "cat", "dog", "tree"]
    return [f"{w}/spk{i:03d}_nohash_0.wav" for w in words for i in range(100)]


def test_label_of() -> None:
    assert label_of("yes/a.wav") == 0
    assert label_of("go/a.wav") == 9
    assert label_of("bed/a.wav") == UNKNOWN


def test_assign_split_follows_official_lists() -> None:
    val, test = {"yes/a.wav"}, {"no/b.wav"}
    assert assign_split("yes/a.wav", val, test) == "val"
    assert assign_split("no/b.wav", val, test) == "test"
    assert assign_split("up/c.wav", val, test) == "train"


def test_select_clips_splits_are_disjoint_and_balanced() -> None:
    utts = _fake_utterances()
    val = set(utts[::10])
    test = set(utts[5::10])
    sel = select_clips(utts, val, test, seed=0)
    paths = {s: {p for p, _ in sel[s]} for s in SPLITS}
    assert not (paths["train"] & paths["val"] or paths["train"] & paths["test"])
    assert not paths["val"] & paths["test"]
    assert paths["val"] <= val and paths["test"] <= test and not paths["train"] & (val | test)
    for s in SPLITS:
        labels = [lab for _, lab in sel[s]]
        n_kw = sum(lab < UNKNOWN for lab in labels)
        assert labels.count(UNKNOWN) == extra_class_size(n_kw)


def test_select_clips_is_deterministic_and_seed_dependent() -> None:
    utts = _fake_utterances()
    a = select_clips(utts, set(), set(), seed=0)
    b = select_clips(utts, set(), set(), seed=0)
    c = select_clips(utts, set(), set(), seed=1)
    assert a == b and a != c


def test_noise_split_holds_out_files_and_segments_do_not_overlap() -> None:
    rng = np.random.default_rng(0)
    names = [*HELD_OUT_NOISE, "a.wav", "b.wav", "c.wav", "d.wav"]
    noise = {n: rng.standard_normal(60 * 16000).astype(np.float32) for n in names}
    segs, held = split_noise(noise)
    assert set(held) == set(HELD_OUT_NOISE)
    for s in SPLITS:
        assert not set(segs[s]) & set(HELD_OUT_NOISE)
        assert set(segs[s]) == {"a.wav", "b.wav", "c.wav", "d.wav"}
    for name in ("a.wav", "b.wav"):
        total = sum(len(segs[s][name]) for s in SPLITS)
        assert total == len(noise[name])
        np.testing.assert_array_equal(
            np.concatenate([segs[s][name] for s in SPLITS]), noise[name]
        )


def test_make_silence_shape_dtype_and_amplitude() -> None:
    rng = np.random.default_rng(0)
    seg = {"a": rng.uniform(-1, 1, 5 * CLIP).astype(np.float32)}
    out = make_silence(seg, 20, rng)
    assert out.shape == (20, CLIP) and out.dtype == np.int16
    assert np.abs(out).max() <= 32767


def test_augment_shift_only() -> None:
    ramp = np.arange(1, CLIP + 1, dtype=np.float32) / CLIP
    x = np.tile(ramp, (50, 1)).astype(np.float32)
    noise = [np.zeros(2 * CLIP, dtype=np.float32)]
    out = augment(x, noise, np.random.default_rng(0), noise_prob=0.0)
    assert out.shape == x.shape
    for i in range(50):
        nz = np.flatnonzero(out[i])
        # A shifted copy of the ramp: contiguous, with at most 1600 zero-filled samples.
        assert (np.diff(nz) == 1).all()
        assert CLIP - len(nz) <= 1600
        np.testing.assert_array_equal(out[i, nz], np.sort(out[i, nz]))


def test_augment_noise_amplitude_and_clipping() -> None:
    x = np.zeros((200, CLIP), dtype=np.float32)
    noise = [np.ones(2 * CLIP, dtype=np.float32)]
    out = augment(x, noise, np.random.default_rng(0), noise_prob=0.8, max_noise_amp=0.1)
    peak = np.max(out, axis=1)
    assert float(np.max(peak)) <= 0.1 + 1e-6
    frac_noisy = (peak > 0).mean()
    assert 0.7 < frac_noisy < 0.9
    loud = augment(np.ones((4, CLIP), dtype=np.float32), noise, np.random.default_rng(1))
    assert float(np.max(loud)) <= 1.0
