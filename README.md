# Handwritten Digit Recognizer

Draw a digit (0-9) on a canvas and a CNN trained on MNIST predicts it, with confidence and per-digit probabilities. Built with Python, TensorFlow/Keras and Streamlit.

## Features
- CNN trained on MNIST (60,000 train / 10,000 test images) with light augmentation
- Drawing canvas with adjustable brush size
- MNIST-style preprocessing: crop, resize to 20x20 keeping aspect ratio, pad to 28x28, center by center of mass
- Empty-canvas detection and low-confidence warning
- Probability bar chart and top-3 predictions
- Evaluation script: accuracy, precision/recall/F1, confusion matrix

## Project structure
```
app.py            Streamlit UI
train.py          Train and save the model
evaluate.py       Test-set metrics and confusion matrix
preprocessing.py  Canvas image -> (1, 28, 28, 1) tensor
requirements.txt
models/           digit_cnn.keras (created by train.py)
reports/          metrics.json, confusion_matrix.png (created by evaluate.py)
```

## Setup (Windows, Python 3.12)
Install the Microsoft Visual C++ Redistributable first (needed by TensorFlow).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python train.py        # downloads MNIST, trains, saves models/digit_cnn.keras
python evaluate.py     # writes reports/metrics.json and confusion_matrix.png
streamlit run app.py
```
Open http://localhost:8501.

Your test accuracy is printed by `train.py` and saved by `evaluate.py`. Results vary slightly between runs, so check your own output.

## Model
Input 28x28x1 -> RandomRotation, RandomTranslation (training only) -> Conv(32) -> MaxPool -> Conv(64) -> MaxPool -> Flatten -> Dropout(0.4) -> Dense(128) -> Dense(10, softmax). Adam optimizer, sparse categorical cross-entropy, early stopping on validation accuracy.

## Tips for better predictions
- Draw one digit, large and centered, with a thick stroke.
- The canvas is white-on-black to match MNIST.

## Deploy on Streamlit Community Cloud
1. Run `train.py` and `evaluate.py`, then commit `models/` and `reports/` (the model file is small).
2. Push to GitHub:
   ```
   git init
   git add .
   git commit -m "Handwritten digit recognizer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/digit-recognizer.git
   git push -u origin main
   ```
3. Go to share.streamlit.io, choose the repo, set the main file to `app.py`, and under Advanced settings select Python 3.12.
4. Deploy. If the build fails, check the logs for package version conflicts.

## Known limitations
- Works best for single, clearly drawn digits. Unusual styles may be misread.
- `streamlit-drawable-canvas` needs Streamlit 1.38.0 or older, hence the pinned version.

## License
MIT
