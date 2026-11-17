# Sign Language Detector

Intro to ML final project. After the Oct 28 meeting the project is an **image classification comparison on the
color [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) dataset** (87,000 images, 29 classes):
traditional ML vs a custom CNN vs transfer learning (ResNet-style and YOLO classification), plus a live webcam demo
that runs the winning model.

Team: Ananyaa Sutaria, Amanda Tantalean, Divya Verma, Nancy Vutuan, Allison Tivvis

## Team rules (short version)
- **One split.** `processed/asl_alphabet_manifest.csv` (owned by Role 1). Train on `train`, tune on `val`, report on `test`.
- **Fixed class order.** `data.CLASSES`: A to Z, then `del`, `nothing`, `space`.
- **One loader.** `from data import load_split` (below). **One seed:** `SEED = 42`.
- Results go to `results/<experiment_name>.json`; every final model exposes `predict(rgb_image) -> (letter, confidence)`.

## Getting the data
Raw images stay out of git (1.2 GB). Download once, outside OneDrive:

```bash
kaggle datasets download -d grassknoted/asl-alphabet -p ~/ml-data/asl-alphabet --unzip
```

Use another location (for example a shared Drive folder on Colab) with the `ASL_DATA_DIR` environment variable,
and `ASL_CACHE_DIR` for the cache the loader builds on first use (about 1 GB).
On Colab, keep the dataset archive in the team's shared Google Drive folder and point
`ASL_DATA_DIR` at its extracted image folder; do not download the archive again for each session.

```bash
python -m venv ~/ml-venv && source ~/ml-venv/bin/activate
pip install -r requirements.txt
```

## Using the shared loader

```python
from data import load_split, class_names, augment, augment_copies, load_real_world, CLASSES

X_train, y_train = load_split("train", size=64, color=True)      # float32 in [0, 1], shape (N, 64, 64, 3)
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

load_split("train", subset="10k")        # 10,005 training images (val/test unchanged): full vs small data
load_split("val", subset="10class")      # only A, B, E, L, M, N, S, V, W, Y with labels 0-9; class_names("10class")
load_split("train", size=224, subset="10k", dtype="uint8")  # sizes above 64 are decoded from the JPEGs
load_split("train", size=28, color=False)                    # grey, shape (N, 28, 28, 1)

X_aug = augment(X_train)                 # training images only; rotate, zoom, shift, brightness, contrast
X_real, y_real = load_real_world()       # photos taken by the team (webcam-style test set)
```

## Notebooks
| File | Role | What |
|---|---|---|
| `01_data.ipynb` | 1 | Exploration, the leak, the new split, cleaning, augmentation, subsets |
| `02_traditional_ml.ipynb` | 2 | Traditional ML features, models, scoring, comparison, and error analysis |
| `03_cnn.ipynb` | 3 | (to come) |
| `04_transfer.ipynb` | 4 | (to come) |
| `05_webcam/` | 5 | (to come) |

## What is in the data work (Role 1)
- **The old random split leaked.** Images are consecutive video frames, so near-identical frames sat in both train and test; a
  1-nearest-neighbour lookup on raw pixels scored 99.7%. The new split cuts each letter into 150-frame blocks, gives every
  split blocks from each lighting condition (dim / normal / bright), and drops 25 frames at every boundary. The same lookup now scores about 45%.
- **Split sizes:** train 54,900 / val 9,225 / test 8,875, with 14,000 buffer frames unused.
- **Blue border:** every original has a 1-3 px blue line on its edges. The loader trims 4 px per side (200 to 192) before resizing.
- **Cleaning:** no unreadable files, no exact duplicates, no empty-frame mislabels. Dark and blurry frames are a real dim recording session,
  so they are flagged (`blur_flag`) and kept; use brightness augmentation.
- **Caveat:** all images show the same hand and room, so test scores will look better than a live webcam. Add photos to the real-world
  test set: `data.ingest_real_world("folder_with_one_subfolder_per_letter", person="yourname")` (JPG/PNG; convert iPhone HEIC first).
- **YOLO classification layout:** `data.make_yolo_folders()` writes `~/ml-data/asl-yolo-cls/{train,val,test}/<letter>/`.

## Experiment 1: Traditional ML

After downloading the dataset and installing `requirements.txt`, run:

```bash
python experiment1.py
```

The script loads the project's fixed splits at 64x64 and extracts grayscale HOG
(9 orientations, 16x16 cells, 2x2 blocks) and eight-bin normalized RGB histograms.
It trains logistic regression, a random forest, and a linear SVM on the full training
set and the manifest's 10,005-image training subset. Both use the same validation
and test sets. The linear models fit their scalers on training features only.
Settings are fixed, not tuned: use validation results for future tuning, not test.

All models use `evaluate.py` to save validation and test JSON reports and prediction
arrays under `results/`. Reports include accuracy, macro and weighted precision,
recall and F1, per-class scores, confusion matrices, training time and warnings.
For test inference timing, 200 images spaced across the test manifest are timed
one at a time on CPU, including feature extraction and prediction but excluding image
loading and resizing. Mean, median and 95th-percentile latency are reported. This
timing is not end-to-end webcam latency. Loading and feature extraction times are
also recorded separately; training features are extracted once for the full set and
reused for the 10k subset.

`results/experiment1/` contains the comparison chart and table, normalized confusion
plots, ranked confusion pairs, prediction paths and error rates by lighting and quality
flags. `results/experiment_log.csv` is the shared run log; its rows are assembled from
the saved evaluation JSON reports. Each run also saves a model artifact exposing
`predict(rgb_image) -> (letter, confidence)`. The wrapper accepts one 64x64 `uint8`
RGB image from the shared preprocessing; SVM confidence is a softmax of decision scores
and is not calibrated. To export wrappers for an existing run without rewriting its
reports, use `python experiment1.py --export-models`.
The dataset has one signer and room, so test scores do not establish webcam performance.
Existing output is not overwritten; use `--results-dir` to choose a new location.
The full run may take substantial time and memory.
The image cache is built in bounded batches and is replaced only after a successful build.
See the completed run's [error analysis](results/experiment1/error_analysis.md).

To rebuild the manifest: `python make_manifest.py` (runs the leak checks and fails if the split is not clean).
