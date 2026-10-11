
"""Train and evaluate a CNN on MNIST, saving the best model and reports."""

import argparse
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Resolve paths relative to this file, not the terminal's current directory.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "digit_cnn.keras"
HISTORY_PATH = REPORT_DIR / "training_history.json"
METRICS_PATH = REPORT_DIR / "test_metrics.json"
CLASSIFICATION_PATH = REPORT_DIR / "classification_report.json"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"


def load_data():
    """Load MNIST and normalize image pixels to the range [0, 1]."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNN input shape: (number of images, 28, 28, 1).
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Build a CNN with data augmentation and regularization."""
    inputs = keras.Input(shape=(28, 28, 1), name="digit_image")

    # Augmentation is active during training, not normal inference.
    x = layers.RandomRotation(
        factor=0.03,
        fill_mode="constant",
        name="random_rotation",
    )(inputs)

    x = layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08,
        fill_mode="constant",
        name="random_translation",
    )(x)

    # Convolution block 1.
    x = layers.Conv2D(
        32, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        32, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

    # Convolution block 2.
    x = layers.Conv2D(
        64, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        64, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    # Classification head.
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(10, activation="softmax", name="digit_probabilities")(x)

    model = keras.Model(inputs, outputs, name="mnist_digit_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_confusion_matrix(y_true, y_pred):
    """Create and save a confusion matrix plot."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(10)))

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")

    ax.set(
        title="MNIST Test Set Confusion Matrix",
        xlabel="Predicted digit",
        ylabel="Actual digit",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )

    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MNIST handwritten digit CNN."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum training epochs. Early stopping may finish sooner.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of images processed per training batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(args.seed)

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()

    print(f"Training images:   {len(x_train)}")
    print(f"Test images:       {len(x_test)}")
    print(f"Image shape:       {x_train.shape[1:]}")
    print(f"Maximum epochs:    {args.epochs}")
    print(f"Batch size:        {args.batch_size}")

    model = build_model()
    model.summary()

    callbacks = [
        # Keep only the model with the best validation accuracy.
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(REPORT_DIR / "training_log.csv"),
            append=False,
        ),
    ]

    print("\nStarting model training...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    # Reload the best checkpoint before the final test evaluation.
    print("\nLoading the best saved model...")
    best_model = keras.models.load_model(MODEL_PATH)

    print("Evaluating on the separate MNIST test set...")
    test_loss, test_accuracy = best_model.evaluate(
        x_test,
        y_test,
        batch_size=args.batch_size,
        verbose=0,
    )

    probabilities = best_model.predict(
        x_test,
        batch_size=args.batch_size,
        verbose=0,
    )
    y_pred = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(10)),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "test_images": int(len(y_test)),
        "epochs_completed": int(len(history.history["loss"])),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
    }

    HISTORY_PATH.write_text(
        json.dumps(history.history, indent=2),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    CLASSIFICATION_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    save_confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 48)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 48)
    print(f"Test accuracy:             {test_accuracy:.4%}")
    print(f"Test loss:                 {test_loss:.4f}")
    print(f"Best validation accuracy:  {metrics['best_validation_accuracy']:.4%}")
    print(f"Epochs completed:          {metrics['epochs_completed']}")

    print("\nFiles saved:")
    for path in (
        MODEL_PATH,
        HISTORY_PATH,
        METRICS_PATH,
        CLASSIFICATION_PATH,
        CONFUSION_MATRIX_PATH,
        REPORT_DIR / "training_log.csv",
    ):
        print(f"  {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
