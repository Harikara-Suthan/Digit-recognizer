"""Train a CNN on MNIST and save it to models/digit_cnn.keras."""
import argparse
import json
from pathlib import Path

import keras
from keras import layers

MODEL_PATH = Path("models/digit_cnn.keras")
HISTORY_PATH = Path("reports/training_history.json")


def load_data():
    """Load MNIST, scale to 0-1 and add the channel dimension."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Small CNN. Augmentation layers only act during training."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.RandomRotation(0.04, fill_mode="constant")(inputs)  # about +/-14 degrees
    x = layers.RandomTranslation(0.1, 0.1, fill_mode="constant")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="digit_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}  Test loss: {test_loss:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    HISTORY_PATH.write_text(json.dumps(history.history, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
