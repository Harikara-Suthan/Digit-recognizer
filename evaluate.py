"""Evaluate the saved model on the MNIST test set.

Writes reports/metrics.json and reports/confusion_matrix.png.
"""
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("models/digit_cnn.keras")
REPORT_DIR = Path("reports")


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} not found. Run `python train.py` first.")

    model = keras.models.load_model(MODEL_PATH)
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print(classification_report(y_test, y_pred, digits=4))

    REPORT_DIR.mkdir(exist_ok=True)
    metrics = {
        "test_accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "test_samples": int(len(y_test)),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (MNIST test set)")
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Saved reports to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
