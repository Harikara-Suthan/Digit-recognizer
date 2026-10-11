"""Streamlit app: draw a digit and get a CNN prediction."""
import json
from pathlib import Path

import keras
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from preprocessing import preprocess_canvas

MODEL_PATH = Path("models/digit_cnn.keras")
METRICS_PATH = Path("reports/metrics.json")
CANVAS_SIZE = 280
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✏️", layout="wide")


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


st.title("Handwritten Digit Recognizer")
st.caption("A CNN trained on MNIST. Draw a digit from 0 to 9 and see the prediction.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first to create `models/digit_cnn.keras`.")
    st.stop()

try:
    model = load_model()
except Exception as exc:  # corrupted file or version mismatch
    st.error(f"Could not load the model: {exc}")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Draw here")
    stroke_width = st.slider("Brush size", 10, 30, 18)
    canvas = st_canvas(
        fill_color="#000000",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        update_streamlit=True,
        key="canvas",
    )
    st.caption("Use the trash icon under the canvas to clear it.")

with right:
    st.subheader("Prediction")
    tensor, preview = (None, None)
    if canvas.image_data is not None:
        tensor, preview = preprocess_canvas(canvas.image_data)

    if tensor is None:
        st.info("The canvas is empty. Draw a digit to get a prediction.")
    else:
        probs = model.predict(tensor, verbose=0)[0]
        order = probs.argsort()[::-1]
        digit, confidence = int(order[0]), float(probs[order[0]])

        c1, c2 = st.columns(2)
        c1.metric("Predicted digit", digit)
        c2.metric("Confidence", f"{confidence:.1%}")
        if confidence < LOW_CONFIDENCE:
            st.warning("Low confidence. Try drawing the digit larger and clearer.")

        st.write("Probability for each digit")
        st.bar_chart(pd.DataFrame({"probability": probs}, index=[str(i) for i in range(10)]))

        st.write("Top 3: " + ", ".join(f"{int(i)} ({probs[i]:.1%})" for i in order[:3]))

        with st.expander("What the model sees (28x28)"):
            st.image(preview, width=140, clamp=True)

if METRICS_PATH.exists():
    with st.expander("Model evaluation (MNIST test set)"):
        metrics = json.loads(METRICS_PATH.read_text())
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{metrics['test_accuracy']:.2%}")
        m2.metric("Macro F1", f"{metrics['macro_f1']:.4f}")
        m3.metric("Test samples", metrics["test_samples"])
        cm_path = Path("reports/confusion_matrix.png")
        if cm_path.exists():
            st.image(str(cm_path), width=450)
