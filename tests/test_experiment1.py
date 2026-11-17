import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import data
import evaluate
import experiment1


class ScoringTests(unittest.TestCase):
    def test_metrics_and_confusion_order(self):
        result = evaluate.evaluate(
            [0, 0, 1, 1, 2], [0, 1, 1, 1, 1], classes=["A", "B", "C"], save=False
        )
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["macro_precision"], 0.5)
        self.assertAlmostEqual(result["macro_recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 4 / 9)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 1, 0]])
        self.assertEqual(evaluate.top_confusions(result)["count"].tolist(), [1, 1])

    def test_invalid_inputs_and_empty_confusions(self):
        for actual, predicted in [([], []), ([0], [2]), ([0], [0, 0])]:
            with self.subTest(actual=actual, predicted=predicted):
                with self.assertRaises(ValueError):
                    evaluate.evaluate(actual, predicted, classes=["A", "B"], save=False)
        with self.assertRaises(ValueError):
            evaluate.time_inference(lambda x: x, [])
        with self.assertRaises(ValueError):
            evaluate.evaluate([0], [0], predict_one=lambda x: x, save=False)
        result = evaluate.evaluate([0], [0], classes=["A", "B"], save=False)
        self.assertEqual(list(evaluate.top_confusions(result).columns), ["true", "predicted", "count"])
        self.assertTrue(evaluate.top_confusions(result).empty)


class FeatureTests(unittest.TestCase):
    def test_histograms_and_hog_distinguish_images(self):
        images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        images[1, :, 32:] = 255
        features = experiment1.extract_features(images)
        self.assertEqual(features.shape, (2, 348))
        np.testing.assert_allclose(features[:, -24:].reshape(2, 3, 8).sum(axis=2), 1)
        self.assertTrue(np.all(features[0, :-24] == 0))
        self.assertGreater(np.linalg.norm(features[1, :-24]), 0)
        with self.assertRaises(ValueError):
            experiment1.extract_features(images.astype(float))

    def test_prediction_interface_returns_letter_and_round_trips(self):
        x = np.eye(348, dtype=np.float32)[:3]
        y = np.arange(3)
        models = [
            LogisticRegression(max_iter=1000).fit(x, y),
            LinearSVC(dual=False, random_state=42).fit(x, y),
        ]
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        for estimator in models:
            predictor = experiment1.TraditionalASLModel(estimator)
            letter, confidence = predictor.predict(image)
            self.assertIn(letter, data.CLASSES)
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.joblib"
                joblib.dump(predictor, path)
                loaded = joblib.load(path)
                self.assertEqual(loaded.predict(image), (letter, confidence))
        with self.assertRaises(ValueError):
            experiment1.TraditionalASLModel(models[0]).predict(np.zeros((200, 200, 3), dtype=np.uint8))


class CacheTests(unittest.TestCase):
    def test_batched_cache_preserves_order_and_failed_rebuild_preserves_existing_cache(self):
        paths = pd.DataFrame({"path": [str(i) for i in range(300)]})

        def fake_read(args):
            path, size = args
            return np.full((size, size, 3), int(path) % 256, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(data, "CACHE_DIR", root), \
                    patch.object(data, "list_images", return_value=paths), \
                    patch.object(data, "_read", side_effect=fake_read):
                cache_path = data.build_cache()
                cache = np.load(cache_path)
                np.testing.assert_array_equal(cache[:, 0, 0, 0], np.arange(300) % 256)
            with patch.object(data, "CACHE_DIR", root), \
                    patch.object(data, "list_images", return_value=paths), \
                    patch.object(data, "_read", side_effect=RuntimeError("decode failed")):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    data.build_cache(force=True)
            np.testing.assert_array_equal(np.load(cache_path), cache)
            self.assertEqual(list(root.iterdir()), [cache_path])


class ExperimentTests(unittest.TestCase):
    def test_complete_run_and_output_protection(self):
        y_train = np.tile(np.arange(3), 4)
        y_test = np.array([0, 1, 2])
        x_train = np.eye(348)[y_train]
        x_test = np.eye(348)[y_test]
        train_manifest = pd.DataFrame({"subset10k": [True] * 6 + [False] * 6})
        test_manifest = pd.DataFrame({
            "y": y_test, "path": ["A/A1.jpg", "B/B1.jpg", "C/C1.jpg"],
            "lighting": ["normal", "dim", "bright"], "blur_flag": [False, True, False],
            "confusable_hash": [False, False, True],
        })

        def fake_load_features(split):
            if split == "train":
                return x_train, y_train, None, 1.0, 2.0
            samples = np.zeros((3, 64, 64, 3), dtype=np.uint8) if split == "test" else None
            return x_test, y_test, samples, 1.0, 2.0

        def models():
            return {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(n_estimators=5, random_state=42),
                "linear_svm": LinearSVC(dual=False, random_state=42),
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / f"rgb{data.CACHE_SIZE}.npy").touch()
            results_dir = root / "results"
            with patch.object(data, "CACHE_DIR", root), \
                    patch.object(data, "load_manifest", side_effect=[train_manifest, test_manifest]), \
                    patch.object(experiment1, "load_features", side_effect=fake_load_features), \
                    patch.object(experiment1, "make_models", side_effect=models):
                experiment1.run(results_dir)
            results = evaluate.load_results(results_dir)
            self.assertEqual(len(results), 12)
            self.assertEqual(sorted(results["set_n_train"].tolist()), [6] * 6 + [12] * 6)
            self.assertTrue((results_dir / "experiment1" / "comparison.png").is_file())
            comparison = pd.read_csv(results_dir / "experiment1" / "comparison.csv")
            self.assertEqual(len(comparison), 6)
            self.assertIn("val_accuracy", comparison.columns)
            log = pd.read_csv(results_dir / "experiment_log.csv")
            self.assertEqual(len(log), 6)
            self.assertEqual(log["owner"].unique().tolist(), ["Divya Verma"])
            self.assertTrue((results_dir / "experiment1_logistic_regression_10k_model.joblib").is_file())
            self.assertTrue((results_dir / "experiment1_random_forest_10k_test_predictions.npz").is_file())
            extra = {column: "" for column in log.columns}
            extra["run_id"] = "another_experiment"
            pd.concat([log, pd.DataFrame([extra])], ignore_index=True).to_csv(
                results_dir / "experiment_log.csv", index=False
            )
            experiment1.save_comparison(results_dir, results_dir / "experiment1")
            refreshed_log = pd.read_csv(results_dir / "experiment_log.csv")
            self.assertIn("another_experiment", refreshed_log["run_id"].tolist())
            self.assertEqual((refreshed_log["run_id"] == "experiment1_logistic_regression_10k").sum(), 1)
            with self.assertRaises(FileExistsError):
                experiment1.run(results_dir)


if __name__ == "__main__":
    unittest.main()
