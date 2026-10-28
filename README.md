# Sign Language Detector

Intro to ML final project. Goal: recognize ASL letters from images, comparing YOLO, a CNN and a Random Forest.

Team: Ananyaa Sutaria, Amanda Tantalean, Divya Verma, Nancy Vutuan, Allison Tivvis

## Data preprocessing

`preprocess_sign_mnist.ipynb` (also available as `preprocess_sign_mnist.py`) preprocesses
[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist):
validates the data, remaps labels to 24 contiguous classes (no J or Z), scales pixels to [0, 1],
and makes a stratified train/val/test split (23,336 / 4,119 / 7,172).

## ASL Alphabet preprocessing

`preprocess_asl_alphabet.ipynb` preprocesses [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
(87,000 200x200 RGB images, 29 classes: A-Z, del, nothing, space). It validates every image, writes a stratified
70/15/15 split manifest (`processed/asl_alphabet_manifest.csv`, 60,900 / 13,050 / 13,050), resizes to 28x28 grayscale
for the Random Forest / CNN baselines, and builds a folder layout for YOLO classification (no bounding boxes exist in this dataset).

The raw images (1.2 GB) are kept outside the project folder; download them with:

```bash
kaggle datasets download -d grassknoted/asl-alphabet -p ~/ml-data/asl-alphabet --unzip
```

Note: the images are consecutive video frames, so near-duplicate frames can fall in different splits and inflate accuracy.

### Run it (Sign Language MNIST)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d datamunge/sign-language-mnist -p data/sign-mnist --unzip
jupyter lab   # open preprocess_sign_mnist.ipynb
```

Raw data and the processed `.npz` are not tracked; the notebook regenerates the `.npz` in `processed/`.
