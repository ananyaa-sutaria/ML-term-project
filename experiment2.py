"""Custom CNN and transfer learning experiments on the fixed ASL Alphabet splits."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import data
import evaluate
from cnn_model import (
    build_simple_cnn,
    build_complex_cnn,
    build_pretrained_cnn,
    CNNASLModel
)

EXPERIMENT_OWNER = "Nancy Vutuan"
BATCH_SIZE = 32
EPOCHS = 10


def make_experiments():
    return {
        "simple_cnn_rgb64": {
            "model": "simple_cnn",
            "size": 64,
            "color": True,
            "subset": None,
            "augmentation": False
        },
        "complex_cnn_rgb64": {
            "model": "complex_cnn",
            "size": 64,
            "color": True,
            "subset": None,
            "augmentation": False
        },
        "complex_cnn_augmented": {
            "model": "complex_cnn",
            "size": 64,
            "color": True,
            "subset": None,
            "augmentation": True
        },
        "complex_cnn_grayscale": {
            "model": "complex_cnn",
            "size": 64,
            "color": False,
            "subset": None,
            "augmentation": False
        },
        "complex_cnn_rgb32": {
            "model": "complex_cnn",
            "size": 32,
            "color": True,
            "subset": None,
            "augmentation": False
        },
        "complex_cnn_10class": {
            "model": "complex_cnn",
            "size": 64,
            "color": True,
            "subset": "10class",
            "augmentation": False
        },
        "pretrained_cnn_rgb64": {
            "model": "pretrained_cnn",
            "size": 64,
            "color": True,
            "subset": None,
            "augmentation": False
        }
    }


def make_model(config):
    channels = 3 if config["color"] else 1
    input_shape = (config["size"], config["size"], channels)
    num_classes = len(data.class_names(config["subset"]))

    builders = {
        "simple_cnn": build_simple_cnn,
        "complex_cnn": build_complex_cnn,
        "pretrained_cnn": build_pretrained_cnn
    }

    return builders[config["model"]](input_shape, num_classes)


def load_data(config):
    datasets = {}

    for split in ("train", "val", "test"):
        images, labels = data.load_split(
            split,
            size=config["size"],
            color=config["color"],
            subset=config["subset"],
            dtype="uint8"
        )

        datasets[split] = (images, labels)

        print(
            f"{split}: {len(images)} images, shape {images.shape}",
            flush=True
        )

    return datasets

def train_model(config, datasets):
    keras.backend.clear_session()
    keras.utils.set_random_seed(data.SEED)

    X_train, y_train = datasets["train"]
    X_val, y_val = datasets["val"]

    if config["augmentation"]:
        print("Applying training augmentation...", flush=True)

        X_train, y_train = data.augment_copies(
            X_train,
            y_train,
            n_copies=1,
            seed=data.SEED
        )

    model = make_model(config)

    print("Training model...", flush=True)
    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    train_time = time.perf_counter() - start

    print(f"Training time: {train_time:.2f} seconds", flush=True)

    return model, history, train_time


def save_model_artifacts(model, history, run_id, results_dir):
    results_dir = Path(results_dir)
    artifact_dir = results_dir / "experiment2"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model.save(results_dir / f"{run_id}_model.keras")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        artifact_dir / f"{run_id}_history.csv",
        index=False
    )

    epochs = range(1, len(history_df) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history_df["accuracy"], label="Training")
    axes[0].plot(epochs, history_df["val_accuracy"], label="Validation")
    axes[0].set_title("Training and validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history_df["loss"], label="Training")
    axes[1].plot(epochs, history_df["val_loss"], label="Validation")
    axes[1].set_title("Training and validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        artifact_dir / f"{run_id}_training_curves.png",
        dpi=150
    )
    plt.close(fig)
    

def evaluate_model(model, config, datasets, run_id,
                   train_time, results_dir):

    classes = data.class_names(config["subset"])
    predictor = CNNASLModel(model, classes)

    settings = {
        "model": config["model"],
        "image_size": config["size"],
        "color": config["color"],
        "subset": config["subset"],
        "augmentation": config["augmentation"],
        "seed": data.SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "adam",
        "n_train": len(datasets["train"][1]),
        "parameters": model.count_params(),
        "tensorflow_version": tf.__version__,
        "inference_device": "CPU",
        "inference_batch_size": 1,
        "model_selection": "final_epoch"
    }

    if config["augmentation"]:
        settings["n_train"] *= 2

    artifact_dir = Path(results_dir) / "experiment2"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        images, labels = datasets[split]

        print(f"Evaluating {run_id} on {split}...", flush=True)

        probabilities = model.predict(
            images,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        predictions = np.argmax(probabilities, axis=1)

        kwargs = {}

        if split == "test":
            rgb_images, _ = data.load_split(
                "test",
                size=64,
                color=True,
                subset=config["subset"],
                dtype="uint8"
            )

            indices = np.linspace(
                0,
                len(rgb_images) - 1,
                min(200, len(rgb_images)),
                dtype=int
            )

            kwargs = {
                "predict_one": predictor.predict,
                "timing_inputs": rgb_images[indices]
            }

        result = evaluate.evaluate(
            labels,
            predictions,
            name=f"{run_id}_{split}",
            settings=settings,
            train_time_s=train_time,
            classes=classes,
            split=split,
            results_dir=results_dir,
            **kwargs
        )

        if split == "test":
            ax = evaluate.plot_confusion(
                result,
                normalize=True
            )

            ax.figure.tight_layout()
            ax.figure.savefig(
                artifact_dir / f"{run_id}_confusion.png",
                dpi=150
            )
            plt.close(ax.figure)

            evaluate.top_confusions(
                result,
                k=20
            ).to_csv(
                artifact_dir / f"{run_id}_confusions.csv",
                index=False
            )


def save_comparison(results_dir):
    results_dir = Path(results_dir)
    artifact_dir = results_dir / "experiment2"

    rows = []
    log_rows = []

    for variant, config in make_experiments().items():
        run_id = f"experiment2_{variant}"

        val_path = results_dir / f"{run_id}_val.json"
        test_path = results_dir / f"{run_id}_test.json"

        if not val_path.exists() or not test_path.exists():
            continue

        val = json.loads(val_path.read_text(encoding="utf-8"))
        test = json.loads(test_path.read_text(encoding="utf-8"))

        settings = test["settings"]

        row = {
            "run_id": run_id,
            "model": config["model"],
            "data_variant": variant,
            "n_train": settings["n_train"],
            "num_classes": len(data.class_names(config["subset"])),
            "val_accuracy": val["accuracy"],
            "test_accuracy": test["accuracy"],
            "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall"],
            "test_macro_f1": test["macro_f1"],
            "train_time_s": test["train_time_s"]
        }

        rows.append(row)

        log_rows.append({
            "run_id": run_id,
            "owner": EXPERIMENT_OWNER,
            "model": config["model"],
            "data_variant": variant,
            "key_settings": json.dumps(settings, sort_keys=True),
            "val_accuracy": row["val_accuracy"],
            "test_accuracy": row["test_accuracy"],
            "test_macro_precision": row["test_macro_precision"],
            "test_macro_recall": row["test_macro_recall"],
            "test_macro_f1": row["test_macro_f1"],
            "train_time_s": row["train_time_s"],
            "notes": "Fixed manifest split; final epoch weights."
        })

    if not rows:
        return

    table = pd.DataFrame(rows)
    table.to_csv(artifact_dir / "comparison.csv", index=False)

    log_path = results_dir / "experiment_log.csv"
    new_log = pd.DataFrame(log_rows)

    if log_path.exists():
        existing_log = pd.read_csv(log_path)

        existing_log = existing_log.loc[
            ~existing_log["run_id"].isin(new_log["run_id"])
        ]

        new_log = pd.concat(
            [existing_log, new_log],
            ignore_index=True
        )

    new_log.to_csv(log_path, index=False)

    print("Comparison table and experiment log saved.", flush=True)
    

def run(results_dir, selected=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = make_experiments()

    for variant, config in experiments.items():

        if selected is not None and variant != selected:
            continue

        run_id = f"experiment2_{variant}"

        model_path = results_dir / f"{run_id}_model.keras"
        val_path = results_dir / f"{run_id}_val.json"
        test_path = results_dir / f"{run_id}_test.json"

        if any(path.exists() for path in (
            model_path, val_path, test_path
        )):
            raise FileExistsError(
                f"Output already exists for {run_id}. "
                "Choose another --results-dir."
            )

        print(f"\nRunning {run_id}", flush=True)
        print(config, flush=True)

        datasets = load_data(config)

        saved_model = results_dir / "experiment2_simple_cnn_model.keras"
        saved_history = results_dir / "experiment2_simple_cnn_history.csv"

        if (
            variant == "simple_cnn_rgb64"
            and saved_model.exists()
            and saved_history.exists()
        ):
            print("Loading previously trained simple CNN...", flush=True)

            model = keras.models.load_model(saved_model)

            history = keras.callbacks.History()
            history.history = pd.read_csv(saved_history).to_dict(
                orient="list"
            )

            train_time = 704.6976

        else:
            model, history, train_time = train_model(
                config,
                datasets
            )

        save_model_artifacts(
            model,
            history,
            run_id,
            results_dir
        )

        evaluate_model(
            model,
            config,
            datasets,
            run_id,
            train_time,
            results_dir
        )

        save_comparison(results_dir)

        del model, history, datasets
        keras.backend.clear_session()

        print(f"Completed {run_id}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=evaluate.RESULTS_DIR
    )

    parser.add_argument(
        "--experiment",
        choices=list(make_experiments()),
        default=None
    )

    args = parser.parse_args()

    run(args.results_dir, args.experiment)