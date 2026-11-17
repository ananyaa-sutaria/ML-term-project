"""Shared data access for the ASL Alphabet project (owned by Role 1).

Everyone loads data through this file so all experiments see the same images,
the same split and the same label order:

    from data import load_split, CLASSES
    X_train, y_train = load_split("train", size=64, color=True)
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

The split lives in processed/asl_alphabet_manifest.csv. Nobody makes their own.
"""
import os
import tempfile
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("ASL_DATA_DIR", Path.home() / "ml-data/asl-alphabet"))
TRAIN_DIR = RAW_DIR / "asl_alphabet_train" / "asl_alphabet_train"
CACHE_DIR = Path(os.environ.get("ASL_CACHE_DIR", Path.home() / "ml-data/cache"))
MANIFEST_PATH = ROOT / "processed" / "asl_alphabet_manifest.csv"
REAL_WORLD_DIR = ROOT / "data" / "real_world"

# Fixed class order: label y means the same thing in every notebook.
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
CLASS_TO_Y = {c: i for i, c in enumerate(CLASSES)}

# The 10-letter subset: five look-alike fists (A, E, M, N, S) plus five distinct shapes.
# Change this list and rerun make_manifest.py to try a different 10.
TEN_CLASSES = ["A", "B", "E", "L", "M", "N", "S", "V", "W", "Y"]

CACHE_SIZE = 64  # the full-dataset cache is built once at this size
EDGE_CROP = 4    # pixels trimmed from every side of the 200x200 originals (blue border artifact)


# ---------------------------------------------------------------- raw images
def list_images():
    """One row per image, ordered by (class, frame number). Frame number is the
    number in the file name (A17.jpg -> 17), i.e. the position in the video."""
    rows = []
    for c in CLASSES:
        for p in (TRAIN_DIR / c).glob("*.jpg"):
            rows.append((f"{c}/{p.name}", c, CLASS_TO_Y[c], int(p.stem[len(c):])))
    df = pd.DataFrame(rows, columns=["path", "label", "y", "frame"])
    return df.sort_values(["y", "frame"]).reset_index(drop=True)


def crop_edge(im):
    """Remove the dataset's blue border line (1-3 px on every image) by trimming EDGE_CROP px per side."""
    return im.crop((EDGE_CROP, EDGE_CROP, im.width - EDGE_CROP, im.height - EDGE_CROP))


def _read(args):
    path, size = args
    with Image.open(TRAIN_DIR / path) as im:
        im = crop_edge(im.convert("RGB"))
        if im.size != (size, size):
            im = im.resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)



def read_images(paths, size, workers=None):
    """Decode and resize images in parallel -> uint8 array (N, size, size, 3).
    Threads, not processes: PIL releases the GIL while decoding, and threads work in a plain
    script, in a notebook and on Colab without a `__main__` guard."""
    with ThreadPool(workers or os.cpu_count()) as pool:
        out = pool.map(_read, [(p, size) for p in paths], chunksize=256)
    return np.stack(out)


def build_cache(force=False):
    """Decode all 87,000 images once at 64x64 RGB (about 1 GB, kept outside the repo)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"rgb{CACHE_SIZE}.npy"
    if f.exists() and not force:
        return f
    np.save(f, read_images(list_images()["path"], CACHE_SIZE))
    paths = list_images()["path"].to_numpy()
    if not len(paths):
        raise ValueError(f"no training images found in {TRAIN_DIR}")
    with tempfile.NamedTemporaryFile(dir=CACHE_DIR, suffix=".npy", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        cache = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.uint8,
            shape=(len(paths), CACHE_SIZE, CACHE_SIZE, 3)
        )
        try:
            with ThreadPool(min(8, os.cpu_count() or 1)) as pool:
                for start in range(0, len(paths), 256):
                    batch = paths[start:start + 256]
                    jobs = [(path, CACHE_SIZE) for path in batch]
                    cache[start:start + len(batch)] = np.stack(pool.map(_read, jobs, chunksize=8))
            cache.flush()
        finally:
            del cache
        os.replace(temp_path, f)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return f


# ------------------------------------------------------------------- loader
def class_names(subset=None):
    """Class names in label order for load_split(..., subset=subset)."""
    return TEN_CLASSES if subset == "10class" else CLASSES


def load_manifest(split=None, subset=None):
    """Manifest rows (optionally one split / subset). The index is the row's position
    in the cache, so it can be used to pull images."""
    m = pd.read_csv(MANIFEST_PATH)
    if split is not None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be 'train', 'val' or 'test'")
        m = m[m["split"] == split]
    if subset == "10k":      # shrinks training only; val and test stay full
        m = m[m["subset10k"] | (m["split"] != "train")]
    elif subset == "10class":  # all splits, only the 10 letters
        m = m[m["ten_class"]]
    elif subset is not None:
        raise ValueError("subset must be None, '10k' or '10class'")
    return m


def _resize(X, size):
    return np.stack([np.asarray(Image.fromarray(x).resize((size, size), Image.LANCZOS)) for x in X])


def _images(m, size):
    if size <= CACHE_SIZE:
        X = np.load(build_cache(), mmap_mode="r")[m.index.to_numpy()]
        return X if size == CACHE_SIZE else _resize(X, size)
    gb = len(m) * size * size * 3 / 1e9
    if gb > 4:
        raise MemoryError(f"{len(m)} images at {size}x{size} RGB is {gb:.1f} GB. "
                          "Use subset='10k' or load_manifest() and stream the files.")
    return read_images(m["path"], size)


def to_gray(X):
    """uint8 RGB (N,H,W,3) -> uint8 grey (N,H,W,1)."""
    g = X[..., 0] * 0.299 + X[..., 1] * 0.587 + X[..., 2] * 0.114
    return np.rint(g).astype(np.uint8)[..., None]


def load_split(split, size=64, color=True, subset=None, dtype="float32"):
    """Images and labels for one split, always the same rows in the same order.

    split   'train', 'val' or 'test'
    size    side length in pixels. 64 and below are fast (cached); larger sizes are decoded
            from the JPEGs, so use them with subset='10k' or for 'val'/'test'.
    color   True -> (N, size, size, 3); False -> grey (N, size, size, 1)
    subset  None (all 29 classes, all training images)
            '10k'     ~10,005 training images (345 per letter, spread over every recording);
                      val and test are unchanged
            '10class' only the 10 letters in TEN_CLASSES, labels 0-9 (see class_names())
    dtype   'float32' scaled to [0, 1], or 'uint8' (0-255, a quarter of the memory)

    Returns X, y. Labels are positions in CLASSES (0-28) unless subset='10class'.
    """
    m = load_manifest(split, subset)
    X = _images(m, size)
    if not color:
        X = to_gray(X)
    y = m["y10" if subset == "10class" else "y"].to_numpy()
    if dtype == "float32":
        X = X.astype(np.float32) / 255.0
    elif dtype != "uint8":
        raise ValueError("dtype must be 'float32' or 'uint8'")
    return X, y


# ------------------------------------------------------------- augmentation
def augment(X, seed=SEED, rotate=15, zoom=(0.9, 1.15), shift=0.08,
            brightness=(0.6, 1.4), contrast=(0.8, 1.2), flip=False):
    """Randomly rotated, zoomed, shifted, brightened copies of a batch of images.

    X is (N, H, W, C) uint8 or float in [0, 1]; the result has the same shape and dtype.
    Apply it to training images only. flip=False because a mirrored hand is a left-hand sign;
    turn it on only if the webcam preview is mirrored.
    """
    rng = np.random.default_rng(seed)
    is_float = X.dtype != np.uint8
    A = np.rint(X * 255).astype(np.uint8) if is_float else X
    n, h, w, ch = A.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    out = np.empty_like(A)
    for i in range(n):
        th = np.deg2rad(rng.uniform(-rotate, rotate))
        s = rng.uniform(*zoom)
        tx, ty = rng.uniform(-shift, shift, 2) * (w, h)
        c, sn = np.cos(th), np.sin(th)
        coef = (c / s, sn / s, cx - (c * (cx + tx) + sn * (cy + ty)) / s,
                -sn / s, c / s, cy - (-sn * (cx + tx) + c * (cy + ty)) / s)
        img = Image.fromarray(A[i, ..., 0] if ch == 1 else A[i])
        fill = tuple(int(v) for v in A[i].mean(axis=(0, 1))) if ch == 3 else int(A[i].mean())
        img = img.transform((w, h), Image.AFFINE, coef, resample=Image.BILINEAR, fillcolor=fill)
        a = np.asarray(img, dtype=np.float32).reshape(h, w, ch)
        a = (a - a.mean()) * rng.uniform(*contrast) + a.mean()   # contrast
        a = a * rng.uniform(*brightness)                          # brightness
        if flip and rng.random() < 0.5:
            a = a[:, ::-1]
        out[i] = np.clip(a, 0, 255).astype(np.uint8)
    return out.astype(np.float32) / 255.0 if is_float else out


def augment_copies(X, y, n_copies=2, seed=SEED, **kw):
    """The originals followed by n_copies augmented versions of every image."""
    Xs, ys = [X], [y]
    for k in range(n_copies):
        Xs.append(augment(X, seed=seed + k + 1, **kw))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


# ------------------------------------------------------ real-world test set
def ingest_real_world(src_dir, person):
    """Copy a teammate's photos into data/real_world/<letter>/<person>_NNN.jpg.

    src_dir must contain one folder per class named exactly like CLASSES (A, B, ..., del,
    nothing, space), each holding that person's photos. HEIC photos must be converted to JPG
    first. Returns a dict of how many photos were added per class.
    """
    added = {}
    for c in CLASSES:
        folder = Path(src_dir) / c
        if not folder.is_dir():
            continue
        dest = REAL_WORLD_DIR / c
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")):
            with Image.open(f) as im:
                im = im.convert("RGB")
                im.thumbnail((800, 800))
                im.save(dest / f"{person}_{n:03d}.jpg", quality=95)
            n += 1
        added[c] = n
    return added


def real_world_files():
    """(path, label) for every real-world photo, in a fixed order."""
    files = sorted(p for c in CLASSES for p in (REAL_WORLD_DIR / c).glob("*.jpg"))
    return [(p, p.parent.name) for p in files]


def load_real_world(size=64, color=True, dtype="float32"):
    """The photos taken by the team, centre-cropped to a square and resized. Same return
    format as load_split. Use it as a second test set that measures webcam-style performance."""
    items = real_world_files()
    if not items:
        raise FileNotFoundError(f"no photos in {REAL_WORLD_DIR}; see ingest_real_world()")
    X = []
    for p, _ in items:
        with Image.open(p) as im:
            im = im.convert("RGB")
            s = min(im.size)
            l, t = (im.width - s) // 2, (im.height - s) // 2
            X.append(np.asarray(im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)))
    X = np.stack(X)
    if not color:
        X = to_gray(X)
    y = np.array([CLASS_TO_Y[c] for _, c in items])
    return (X.astype(np.float32) / 255.0 if dtype == "float32" else X), y


# ------------------------------------------------------------- YOLO folders
def make_yolo_folders(dest=Path.home() / "ml-data" / "asl-yolo-cls"):
    """train/ val/ test/ folders (one subfolder per class) following the manifest, for YOLO
    classification. Holds edge-cropped 192x192 JPEG copies so YOLO sees the same pixels as every
    other model (about 1 GB, kept outside the repo). Rebuilds dest from scratch; buffer frames are
    left out. Only deletes dest if it was created by this function."""
    import shutil
    dest = Path(dest)
    if dest.exists():
        if not (dest / ".made_by_make_yolo_folders").exists():
            raise RuntimeError(f"{dest} was not created by make_yolo_folders(); refusing to delete it")
        shutil.rmtree(dest)
    m = load_manifest()
    jobs = [(path, dest / split / label / Path(path).name)
            for path, label, split in m.loc[m["split"] != "buffer", ["path", "label", "split"]].itertuples(index=False)]
    for d in {j[1].parent for j in jobs}:
        d.mkdir(parents=True, exist_ok=True)

    def save(job):
        src, out = job
        with Image.open(TRAIN_DIR / src) as im:
            crop_edge(im.convert("RGB")).save(out, quality=95)

    with ThreadPool(os.cpu_count()) as pool:
        pool.map(save, jobs, chunksize=256)
    (dest / ".made_by_make_yolo_folders").write_text("safe to delete\n")
    return dest
