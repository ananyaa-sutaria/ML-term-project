"""Image quality scan for the ASL Alphabet data (Role 1).

quality_scan() reads every original 200x200 image once and measures:
  brightness   mean grey level (0-255)
  contrast     grey-level standard deviation
  sharpness    variance of the Laplacian (low = blurry)
  md5          hash of the file bytes (exact duplicates)
  dhash        64-bit perceptual hash (near duplicates)
The results are saved to processed/asl_quality.csv.
"""
import hashlib
import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import data

QUALITY_PATH = data.ROOT / "processed" / "asl_quality.csv"


def _measure(path):
    f = data.TRAIN_DIR / path
    raw = f.read_bytes()
    with Image.open(f) as im:
        g = data.crop_edge(im).convert("L")  # without the blue border line
        a = np.asarray(g, dtype=np.float32)
        small = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    dhash = int("".join("1" if b else "0" for b in bits), 2)
    return (path, a.mean(), a.std(), ndimage.laplace(a).var(),
            hashlib.md5(raw).hexdigest(), dhash)


def quality_scan(force=False):
    if QUALITY_PATH.exists() and not force:
        return pd.read_csv(QUALITY_PATH, dtype={"dhash": "uint64"})
    paths = data.list_images()["path"].tolist()
    with ThreadPool(os.cpu_count()) as pool:
        rows = pool.map(_measure, paths, chunksize=256)
    q = pd.DataFrame(rows, columns=["path", "brightness", "contrast", "sharpness", "md5", "dhash"])
    q["dhash"] = q["dhash"].astype("uint64")
    q.to_csv(QUALITY_PATH, index=False)
    return q
