"""CNN architectures and webcam-facing wrapper for ASL classification."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

import data


def build_simple_cnn(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_complex_cnn(input_shape, num_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_pretrained_cnn(input_shape, num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class CNNASLModel:
    """Expose a trained CNN using the shared prediction interface."""

    def __init__(self, model, classes=None):
        self.model = model
        self.classes = tuple(data.CLASSES if classes is None else classes)

    def predict(self, rgb_image):
        """Return (letter, confidence) for one preprocessed RGB image."""
        image = np.asarray(rgb_image)

        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError(
                "rgb_image must be uint8 RGB with shape (64, 64, 3)"
            )

        height, width, channels = self.model.input_shape[1:]

        if image.shape[:2] != (height, width):
            image = np.asarray(
                Image.fromarray(image).resize(
                    (width, height), Image.LANCZOS
                )
            )

        if channels == 1:
            image = data.to_gray(image[None, ...])[0]

        with tf.device("/CPU:0"):
            scores = self.model(
                image[None, ...], training=False
            ).numpy()[0]

        predicted = int(np.argmax(scores))
        confidence = float(scores[predicted])

        return self.classes[predicted], confidence