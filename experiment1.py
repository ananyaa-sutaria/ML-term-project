"""Traditional ML on the project's fixed ASL Alphabet splits."""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import sklearn
import skimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

import data
import evaluate
from traditional_model import TraditionalASLModel, extract_features

EXPERIMENT_OWNER = "Divya Verma"


def make_models():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=data.SEED)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=data.SEED, n_jobs=4
        ),
        "linear_svm": make_pipeline(
            StandardScaler(), LinearSVC(dual=False, max_iter=5000, random_state=data.SEED)
        ),
    }


def load_features(split):
    start = time.perf_counter()
    images, labels = data.load_split(split, size=64, color=True, dtype="uint8")
    load_seconds = time.perf_counter() - start
    print(f"Loaded {len(images)} {split} images in {load_seconds:.1f}s; extracting features...", flush=True)
    timing_images = None
    if split == "test":
        indices = np.linspace(0, len(images) - 1, min(200, len(images)), dtype=int)
        timing_images = images[indices]
    start = time.perf_counter()
    if not len(images):
        raise ValueError(f"{split} split is empty")
    features = np.empty((len(images), 348), dtype=np.float32)
    for offset in range(0, len(images), 5000):
        end = min(offset + 5000, len(images))
        features[offset:end] = extract_features(images[offset:end])
        print(f"{split} features: {end}/{len(images)}", flush=True)
    feature_seconds = time.perf_counter() - start
    return features, labels, timing_images, load_seconds, feature_seconds


def save_error_analysis(artifact_dir, name, manifest, labels, predictions, result):
    rows = manifest[["path", "lighting", "blur_flag", "confusable_hash"]].copy()
    rows["actual"] = [data.CLASSES[i] for i in labels]
    rows["predicted"] = [data.CLASSES[i] for i in predictions]
    rows["error"] = labels != predictions
    rows.to_csv(artifact_dir / f"{name}_predictions.csv", index=False)
    evaluate.top_confusions(result, k=len(data.CLASSES) ** 2).to_csv(
        artifact_dir / f"{name}_confusions.csv", index=False
    )
    groups = []
    for column in ("lighting", "blur_flag", "confusable_hash"):
        for value, group in rows.groupby(column, dropna=False):
            groups.append({"condition": column, "value": str(value), "count": len(group),
                           "errors": int(group["error"].sum()),
                           "error_rate": float(group["error"].mean())})
    pd.DataFrame(groups).to_csv(artifact_dir / f"{name}_conditions.csv", index=False)
    ax = evaluate.plot_confusion(result, normalize=True)
    ax.figure.tight_layout()
    ax.figure.savefig(artifact_dir / f"{name}_confusion.png", dpi=150)
    plt.close(ax.figure)


def save_comparison(results_dir, artifact_dir):
    results = evaluate.load_results(results_dir)
    results = results[results["name"].str.startswith("experiment1_")]
    test = results[results["split"] == "test"]
    val = results[results["split"] == "val"]
    if len(test) != 6 or len(val) != 6:
        raise ValueError("expected six validation and six test reports for Experiment 1")
    val_accuracy = val.set_index(["set_model", "set_training_size"])["accuracy"]
    table = test.copy()
    table["val_accuracy"] = [
        val_accuracy.loc[(row.set_model, row.set_training_size)]
        for row in table.itertuples(index=False)
    ]
    table = table.rename(columns={
        "set_model": "model", "set_training_size": "training_size", "set_n_train": "n_train",
        "accuracy": "test_accuracy", "macro_precision": "test_macro_precision",
        "macro_recall": "test_macro_recall", "macro_f1": "test_macro_f1",
        "ms_per_image": "inference_ms",
    })
    model_order = {name: index for index, name in enumerate(make_models())}
    table["_model_order"] = table["model"].map(model_order)
    table["_size_order"] = table["training_size"].map({"10k": 0, "full": 1})
    table = table.sort_values(["_size_order", "_model_order"]).drop(
        columns=["_model_order", "_size_order"]
    )
    table.to_csv(artifact_dir / "comparison.csv", index=False)

    log_rows = []
    for row in table.itertuples(index=False):
        run_id = f"experiment1_{row.model}_{row.training_size}"
        report = json.loads((results_dir / f"{run_id}_test.json").read_text(encoding="utf-8"))
        settings = report["settings"]
        log_rows.append({
            "run_id": run_id,
            "owner": EXPERIMENT_OWNER,
            "model": row.model,
            "data_variant": row.training_size,
            "key_settings": json.dumps({
                "estimator": settings["estimator"],
                "features": settings["features"],
                "seed": settings["seed"],
                "n_train": settings["n_train"],
            }, sort_keys=True),
            "val_accuracy": row.val_accuracy,
            "test_accuracy": row.test_accuracy,
            "test_macro_precision": row.test_macro_precision,
            "test_macro_recall": row.test_macro_recall,
            "test_macro_f1": row.test_macro_f1,
            "train_time_s": row.train_time_s,
            "notes": "Fixed manifest split; no hyperparameter tuning.",
        })
    log_path = results_dir / "experiment_log.csv"
    new_log = pd.DataFrame(log_rows)
    if log_path.is_file():
        existing_log = pd.read_csv(log_path)
        existing_log = existing_log.loc[~existing_log["run_id"].isin(new_log["run_id"])]
        new_log = pd.concat([existing_log, new_log], ignore_index=True)
    new_log.to_csv(log_path, index=False)

    table = table.rename(columns={
        "test_accuracy": "accuracy",
        "test_macro_precision": "macro_precision",
        "test_macro_recall": "macro_recall",
        "test_macro_f1": "macro_f1",
    })
    labels = table["model"] + "\n" + table["training_size"]
    positions = np.arange(len(table))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for offset, (column, label) in enumerate((
        ("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")
    )):
        axes[0].bar(positions + (offset - 1.5) * 0.2, table[column], width=0.2, label=label)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Test metrics")
    axes[0].legend()
    for ax, column, title in zip(
        axes[1:], ("train_time_s", "inference_ms"),
        ("Training time (seconds)", "CPU batch-1 inference (ms/image)")
    ):
        ax.bar(labels, table[column])
        ax.set_title(title)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(artifact_dir / "comparison.png", dpi=150)
    plt.close(fig)


def export_models(results_dir):
    """Refit the six deterministic models and export wrappers matching saved test predictions."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {results_dir}")
    model_paths = [
        results_dir / f"experiment1_{name}_{size}_model.joblib"
        for size in ("10k", "full") for name in make_models()
    ]
    if any(path.exists() for path in model_paths):
        raise FileExistsError("one or more Experiment 1 model artifacts already exist")

    train_features, train_labels, _, _, _ = load_features("train")
    test_features, test_labels, test_images, _, _ = load_features("test")
    train_manifest = data.load_manifest("train")
    subset_mask = train_manifest["subset10k"].to_numpy(dtype=bool)
    if len(subset_mask) != len(train_labels) or not np.array_equal(
        data.load_manifest("test")["y"].to_numpy(), test_labels
    ):
        raise ValueError("loaded images do not match the fixed manifest")

    for size, mask in (("10k", subset_mask), ("full", slice(None))):
        x_train, y_train = train_features[mask], train_labels[mask]
        for name, estimator in make_models().items():
            run_id = f"experiment1_{name}_{size}"
            prediction_path = results_dir / f"{run_id}_test_predictions.npz"
            if not prediction_path.is_file():
                raise FileNotFoundError(f"saved test predictions are missing: {prediction_path}")
            print(f"Exporting {name} ({size}, {len(y_train)} images)...", flush=True)
            with threadpool_limits(limits=1):
                estimator.fit(x_train, y_train)
            expected = np.load(prediction_path)["y_pred"]
            actual = estimator.predict(test_features)
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"refitted {run_id} predictions do not match its saved report")
            model = TraditionalASLModel(estimator)
            letter, confidence = model.predict(test_images[0])
            if letter != data.CLASSES[int(actual[0])] or not 0 <= confidence <= 1:
                raise RuntimeError(f"prediction interface verification failed for {run_id}")
            joblib.dump(model, results_dir / f"{run_id}_model.joblib", compress=3)


def run(results_dir):
    results_dir = Path(results_dir)
    artifact_dir = results_dir / "experiment1"
    names = [f"experiment1_{model}_{size}_{split}"
             for model in make_models() for size in ("10k", "full") for split in ("val", "test")]
    existing = [results_dir / f"{name}{suffix}" for name in names for suffix in (".json", "_predictions.npz")
                if (results_dir / f"{name}{suffix}").exists()]
    existing.extend(
        results_dir / f"experiment1_{model}_{size}_model.joblib"
        for model in make_models() for size in ("10k", "full")
        if (results_dir / f"experiment1_{model}_{size}_model.joblib").exists()
    )
    if artifact_dir.exists() or existing:
        raise FileExistsError("Experiment 1 output already exists; choose a different --results-dir")
    cache = data.CACHE_DIR / f"rgb{data.CACHE_SIZE}.npy"
    if not cache.is_file() and not data.TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"No image cache at {cache} or raw training images at {data.TRAIN_DIR}. "
            "Download the dataset or set ASL_DATA_DIR / ASL_CACHE_DIR."
        )
    if not cache.is_file():
        raw_paths = data.list_images()["path"].to_numpy()
        manifest_paths = data.load_manifest()["path"].to_numpy()
        if not np.array_equal(raw_paths, manifest_paths):
            raise ValueError("raw image paths do not match the manifest; refusing to build a misaligned cache")
    artifact_dir.mkdir(parents=True)
    train_manifest = data.load_manifest("train")
    test_manifest = data.load_manifest("test").reset_index(drop=True)
    features, labels, timing = {}, {}, {}
    timing_images = None
    for split in ("train", "val", "test"):
        print(f"Loading and extracting {split} features...", flush=True)
        features[split], labels[split], samples, load_time, feature_time = load_features(split)
        timing[split] = {"load_seconds": load_time, "feature_extraction_seconds": feature_time,
                         "n_samples": len(labels[split])}
        if split == "test":
            timing_images = samples
    small_mask = train_manifest["subset10k"].to_numpy(dtype=bool)
    if len(small_mask) != len(labels["train"]) or not small_mask.any():
        raise ValueError("10k subset does not match the training features")
    if len(test_manifest) != len(labels["test"]) or not np.array_equal(
        test_manifest["y"].to_numpy(), labels["test"]
    ):
        raise ValueError("test manifest does not match the test features")

    for size, mask in (("10k", small_mask), ("full", slice(None))):
        x_train, y_train = features["train"][mask], labels["train"][mask]
        for name, model in make_models().items():
            print(f"Training {name} ({size}, {len(y_train)} images)...", flush=True)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x_train, y_train)
            train_time = time.perf_counter() - start
            fit_warnings = [str(w.message) for w in caught]
            for warning in fit_warnings:
                print(f"Training warning: {warning}", flush=True)
            settings = {
                "model": name, "training_size": size, "n_train": len(y_train),
                "seed": data.SEED, "features": "64x64 grayscale HOG (9 orientations, 16x16 cells, "
                                               "2x2 blocks) and eight-bin RGB histograms",
                "estimator": repr(model), "scikit_learn_version": sklearn.__version__,
                "scikit_image_version": skimage.__version__,
                "inference_device": "CPU", "inference_batch_size": 1,
                "data_timing": timing, "fit_warnings": fit_warnings,
                "inference_scope": "64x64 RGB feature extraction and model prediction; excludes image loading and resizing",
            }
            model_path = results_dir / f"experiment1_{name}_{size}_model.joblib"
            if model_path.exists():
                raise FileExistsError(f"model artifact already exists: {model_path}")
            predictor = TraditionalASLModel(model)
            joblib.dump(predictor, model_path, compress=3)
            for split in ("val", "test"):
                predictions = model.predict(features[split])
                result_name = f"experiment1_{name}_{size}_{split}"
                kwargs = {}
                if split == "test":
                    kwargs = {
                        "predict_one": predictor.predict,
                        "timing_inputs": timing_images,
                    }
                result = evaluate.evaluate(
                    labels[split], predictions, name=result_name, settings=settings,
                    train_time_s=train_time, split=split, results_dir=results_dir, **kwargs
                )
                if split == "test":
                    save_error_analysis(artifact_dir, f"{name}_{size}", test_manifest,
                                        labels[split], predictions, result)
    save_comparison(results_dir, artifact_dir)
    print(f"Saved results to {results_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=evaluate.RESULTS_DIR)
    parser.add_argument("--export-models", action="store_true",
                        help="refit models and export joblib wrappers verified against saved test predictions")
    args = parser.parse_args()
    if args.export_models:
        export_models(args.results_dir)
    else:
        run(args.results_dir)
