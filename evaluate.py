"""Shared scoring for every experiment (starter written by Role 1 for Role 2 to own and extend).

    from evaluate import evaluate
    res = evaluate(y_test, y_pred, predict_one=model_predict_one, timing_inputs=X_test[:200],
                   name="rf_hog_full", settings={"model": "RandomForest", "n_estimators": 300},
                   train_time_s=412.0)

Every experiment calls this so nobody retypes numbers. It returns the metrics and saves
    results/<name>.json              metrics, settings, training time
    results/<name>_predictions.npz   the test predictions (y_true, y_pred)
The comparison charts are built from those files with load_results().

Inference time is measured one way for everyone: milliseconds per image, batch size 1, on CPU.
Run the timing on CPU yourself (e.g. torch.set_num_threads(...) and model.to("cpu"), or
CUDA_VISIBLE_DEVICES=""); this file cannot force it.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from data import CLASSES, ROOT

RESULTS_DIR = ROOT / "results"


def time_inference(predict_one, inputs, warmup=5):
    """Milliseconds per image with batch size 1. predict_one gets ONE input at a time, exactly
    as the webcam app would call it (e.g. one image, or one feature vector)."""
    inputs = list(inputs)
    if not inputs:
        raise ValueError("timing inputs must not be empty")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    for x in inputs[:warmup]:
        predict_one(x)
    times = []
    for x in inputs:
        t = time.perf_counter()
        predict_one(x)
        times.append((time.perf_counter() - t) * 1000)
    return {"mean": float(np.mean(times)), "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)), "n": len(times)}


def evaluate(y_true, y_pred, predict_one=None, timing_inputs=None, name=None, settings=None,
             train_time_s=None, classes=CLASSES, split="test", results_dir=RESULTS_DIR, save=True):
    """Score predictions and (optionally) time the model.

    y_true, y_pred   integer labels, positions in `classes` (data.CLASSES unless you used
                     subset='10class', then pass classes=data.TEN_CLASSES)
    predict_one      function taking a single input, for the timing; skip to leave timing out
    timing_inputs    ~200 single inputs to time with
    name             experiment name; the results file is results/<name>.json
    settings         dict of whatever defines the run (model, data variant, size, augmentation, ...)
    train_time_s     seconds spent training
    Returns the metrics dict (also written to disk when save=True and name is given).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if (y_true.ndim != 1 or y_pred.ndim != 1 or not len(y_true)
            or y_true.shape != y_pred.shape):
        raise ValueError("y_true and y_pred must be nonempty, equally sized 1-D arrays")
    labels = list(range(len(classes)))
    if not labels or not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError("labels must be indices into classes")
    if (predict_one is None) != (timing_inputs is None):
        raise ValueError("predict_one and timing_inputs must be supplied together")
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    res = {
        "name": name, "split": split, "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]), "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(), "classes": list(classes),
        "inference_ms_per_image": (time_inference(predict_one, timing_inputs)
                                   if predict_one is not None and timing_inputs is not None else None),
        "train_time_s": train_time_s, "settings": settings or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if save and name:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(res, indent=2))
        np.savez_compressed(out / f"{name}_predictions.npz", y_true=y_true, y_pred=y_pred)
    return res


def load_results(results_dir=RESULTS_DIR):
    """One row per saved experiment, for the comparison table and charts."""
    rows = []
    for f in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(f.read_text())
        t = r.get("inference_ms_per_image") or {}
        rows.append({"name": r["name"], "split": r["split"], "accuracy": r["accuracy"],
                     "macro_precision": r["macro_precision"], "macro_recall": r["macro_recall"],
                     "macro_f1": r["macro_f1"], "ms_per_image": t.get("mean"),
                     "train_time_s": r["train_time_s"], **{f"set_{k}": v for k, v in r["settings"].items()}})
    return pd.DataFrame(rows)


def top_confusions(res, k=10):
    """The k most frequent (true letter, predicted letter) mistakes, for the error analysis."""
    cm = np.array(res["confusion_matrix"])
    np.fill_diagonal(cm, 0)
    if k < 0:
        raise ValueError("k must be non-negative")
    order = np.argsort(-cm, axis=None, kind="stable")
    order = [i for i in order if cm.flat[i] > 0][:k]
    classes = res["classes"]
    return pd.DataFrame([{"true": classes[i // len(classes)], "predicted": classes[i % len(classes)],
                          "count": int(cm.flat[i])} for i in order],
                        columns=["true", "predicted", "count"])


def plot_confusion(res, normalize=True, ax=None):
    """Confusion matrix heatmap (rows = true letter). normalize=True shows recall per letter."""
    import matplotlib.pyplot as plt
    cm = np.array(res["confusion_matrix"], dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cm, cmap="Blues")
    n = len(res["classes"])
    ax.set_xticks(range(n)); ax.set_xticklabels(res["classes"], rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(res["classes"], fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(res["name"] or "confusion matrix")
    return ax


if __name__ == "__main__":  # quick self-check on a small, fast model: python evaluate.py
    import tempfile

    from sklearn.linear_model import LogisticRegression

    from data import load_split

    Xtr, ytr = load_split("train", size=28, color=False, subset="10k")
    Xte, yte = load_split("test", size=28, color=False)
    flat = lambda X: X.reshape(len(X), -1)
    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=200).fit(flat(Xtr), ytr)
    train_s = time.perf_counter() - t0
    pred = clf.predict(flat(Xte))
    with tempfile.TemporaryDirectory() as d:
        res = evaluate(yte, pred, predict_one=lambda x: clf.predict(x.reshape(1, -1)), timing_inputs=flat(Xte[:50]),
                       name="selfcheck_logreg_28gray_10k", settings={"model": "LogReg", "size": 28, "subset": "10k"},
                       train_time_s=train_s, results_dir=d)
        print(load_results(d).T)
    print(top_confusions(res, 5))
