"""Shared HOG/color features and webcam-facing wrapper for traditional ASL models."""

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog

import data


def extract_features(images):
    """Extract grayscale HOG and normalized eight-bin RGB histograms."""
    if images.ndim != 4 or images.shape[1:] != (64, 64, 3) or images.dtype != np.uint8:
        raise ValueError("expected uint8 images with shape (N, 64, 64, 3)")
    if not len(images):
        raise ValueError("images must not be empty")
    features = np.empty((len(images), 348), dtype=np.float32)
    for i, image in enumerate(images):
        features[i, :324] = hog(
            rgb2gray(image), orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), block_norm="L2-Hys"
        )
        for channel in range(3):
            start = 324 + channel * 8
            features[i, start:start + 8] = (
                np.histogram(image[..., channel], bins=8, range=(0, 256))[0] / (64 * 64)
            )
    return features


class TraditionalASLModel:
    """Expose a fitted classifier for 64x64 RGB images from the shared preprocessing."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.classes = tuple(data.CLASSES)

    def predict(self, rgb_image):
        """Return (letter, confidence) for one preprocessed uint8 RGB image.

        SVM confidence is a softmax of decision scores, not a calibrated probability.
        """
        image = np.asarray(rgb_image)
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError("rgb_image must be uint8 RGB with shape (64, 64, 3) from data.load_split")
        features = extract_features(image[None, ...])
        predicted = int(self.estimator.predict(features)[0])
        if not 0 <= predicted < len(self.classes):
            raise ValueError("estimator returned a label outside data.CLASSES")
        if hasattr(self.estimator, "predict_proba"):
            scores = self.estimator.predict_proba(features)[0]
            estimator_classes = list(self.estimator.classes_)
            confidence = float(scores[estimator_classes.index(predicted)])
        elif hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(features)).reshape(-1)
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            confidence = float(probabilities[predicted])
        else:
            raise TypeError("estimator must implement predict_proba or decision_function")
        return self.classes[predicted], confidence
