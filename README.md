# Sign Language Detector

Intro to ML final project. Goal: recognize ASL letters from images, comparing YOLO, a CNN and a Random Forest.

Team: Ananyaa Sutaria, Amanda Tantalean, Divya Verma, Nancy Vutuan, Allison Tivvis

## Data preprocessing

`preprocess_sign_mnist.ipynb` (also available as `preprocess_sign_mnist.py`) preprocesses
[Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist):
validates the data, remaps labels to 24 contiguous classes (no J or Z), scales pixels to [0, 1],
and makes a stratified train/val/test split (23,336 / 4,119 / 7,172).

### Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d datamunge/sign-language-mnist -p data/sign-mnist --unzip
jupyter lab   # open preprocess_sign_mnist.ipynb
```

Raw data and the processed `.npz` are not tracked; the notebook regenerates the `.npz` in `processed/`.
