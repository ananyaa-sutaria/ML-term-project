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
| `02_traditional_ml.ipynb` | 2 | (to come) |
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

To rebuild the manifest: `python make_manifest.py` (runs the leak checks and fails if the split is not clean).

## Older work
`archive/` holds the first preprocessing pass (Sign Language MNIST and a 28x28 ASL notebook with a random split). It is superseded by the files above.
