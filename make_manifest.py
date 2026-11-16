"""Build processed/asl_alphabet_manifest.csv: the one split everybody uses (Role 1).

Why not a random split: every letter is a continuous video, so neighbouring frames are
near-duplicates. A random split puts almost the same picture in train and test (a 1-nearest-
neighbour lookup on raw pixels scores 99.7% on it). Instead we split by *time*:

  1. Cut each letter's 3,000 frames into 20 blocks of 150 consecutive frames.
  2. Recording conditions repeat across letters (frames ~1950-2400 are dim for almost every
     letter, ~1050-1950 are bright), so blocks are grouped into dim / normal / bright regimes.
  3. Per letter, val gets one random block from each regime, test gets one from each, train gets
     the other 14 blocks. Every split therefore sees every lighting condition.
  4. 25 frames on each side of every train/val/test boundary are dropped (split = "buffer"),
     because frames only stop looking alike about 25-50 frames apart.

Also adds the 10k-image training subset and the 10-letter subset, plus quality flags.

    python make_manifest.py          # rebuild the CSV and print the checks
"""
import numpy as np
import pandas as pd

import clean
import data

BLOCK = 150
BUFFER = 25
N_BLOCKS = 3000 // BLOCK
SUBSET_PER_CLASS = 345  # 29 x 345 = 10,005 training images ("10k")
BLUR_Z = -3.5


def block_regimes(q):
    """dim / normal / bright for each block, from the median brightness across letters."""
    t = q.groupby(["y", "block"])["brightness"].mean().unstack()
    rel = t.sub(t.mean(axis=1), axis=0).median()
    return rel.map(lambda r: "dim" if r < -10 else "bright" if r > 10 else "normal")


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    return (s - med) / mad


def build_manifest():
    df = data.list_images()
    q = clean.quality_scan().set_index("path").loc[df["path"]].reset_index()
    df["block"] = (df["frame"] - 1) // BLOCK
    q["y"], q["block"] = df["y"], df["block"]
    regime = block_regimes(q)
    df["lighting"] = df["block"].map(regime)

    # ---- time-block split, one val and one test block per lighting regime
    assign = np.empty((29, N_BLOCKS), dtype=object)
    for c in range(29):
        rng = np.random.default_rng(data.SEED + c)
        assign[c, :] = "train"
        for r in ["dim", "normal", "bright"]:
            blocks = rng.permutation(regime.index[regime == r].to_numpy())
            assign[c, blocks[0]], assign[c, blocks[1]] = "val", "test"
    split = assign[df["y"], df["block"]].copy()

    # ---- buffer frames next to a block that belongs to a different split
    pos = (df["frame"] - 1) % BLOCK
    for c in range(29):
        for j in range(1, N_BLOCKS):
            if assign[c, j] != assign[c, j - 1]:
                m = df["y"] == c
                split[(m & (df["block"] == j) & (pos < BUFFER)).to_numpy()] = "buffer"
                split[(m & (df["block"] == j - 1) & (pos >= BLOCK - BUFFER)).to_numpy()] = "buffer"
    df["split"] = split

    # ---- subsets (the 10k subset only shrinks *training*; val/test stay the same)
    df["subset10k"] = False
    for c in range(29):
        idx = df.index[(df["y"] == c) & (df["split"] == "train")]
        keep = np.linspace(0, len(idx) - 1, SUBSET_PER_CLASS).round().astype(int)
        df.loc[idx[keep], "subset10k"] = True
    df["ten_class"] = df["label"].isin(data.TEN_CLASSES)
    df["y10"] = df["label"].map({c: i for i, c in enumerate(data.TEN_CLASSES)}).fillna(-1).astype(int)

    # ---- quality flags (reported, not removed)
    z = q.assign(lighting=df["lighting"]).groupby(["y", "lighting"])["sharpness"].transform(robust_z)
    df["blur_flag"] = z < BLUR_Z
    shared = q.assign(y=df["y"]).groupby("dhash")["y"].transform("nunique") > 1
    df["confusable_hash"] = shared  # same perceptual hash as an image of another letter
    return df


def verify(m):
    """Raise if the split is not leak-free or not balanced the way the plan says."""
    assert list(m["label"].unique()) == data.CLASSES, "class order changed"
    assert not m["path"].duplicated().any()
    for c in range(29):
        f = m[m["y"] == c]
        tr = f.loc[f["split"] == "train", "frame"].to_numpy()
        for s in ["val", "test"]:
            other = f.loc[f["split"] == s, "frame"].to_numpy()
            gap = np.abs(other[:, None] - tr[None, :]).min()
            assert gap > BUFFER, f"class {c}: {s} frame only {gap} frames from a train frame"
        for s in ["train", "val", "test"]:
            assert set(f.loc[f["split"] == s, "lighting"]) == {"dim", "normal", "bright"}, (c, s)
    assert m.loc[m["subset10k"], "split"].eq("train").all()
    return True


if __name__ == "__main__":
    m = build_manifest()
    verify(m)
    m.drop(columns=[]).to_csv(data.MANIFEST_PATH, index=False)
    print(m["split"].value_counts().to_string())
    print("10k subset:", int(m["subset10k"].sum()), "| 10-class rows:", int(m["ten_class"].sum()),
          "| blur flags:", int(m["blur_flag"].sum()), "| confusable-hash images:", int(m["confusable_hash"].sum()))
    print("saved", data.MANIFEST_PATH)
