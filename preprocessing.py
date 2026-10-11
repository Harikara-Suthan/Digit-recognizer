"""Convert a user-drawn canvas image into an MNIST-style model input.

MNIST digits are white strokes on a black background, scaled to fit a
20x20 box and centered by center of mass inside a 28x28 image.
We reproduce those steps so drawn digits look like the training data.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

THRESHOLD = 30  # gray values below this are treated as background
MIN_PIXELS = 20  # fewer stroke pixels than this counts as an empty canvas


def to_grayscale(image_data) -> np.ndarray:
    """Return a 2D float array (0-255) with white strokes on black.

    Accepts the RGBA array returned by streamlit-drawable-canvas.
    """
    arr = np.asarray(image_data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    gray = arr[..., :3].mean(axis=2)
    if arr.shape[2] == 4:  # blend with alpha so transparent pixels become black
        gray = gray * (arr[..., 3] / 255.0)
    return gray


def preprocess_canvas(image_data, invert: bool = False):
    """Prepare a canvas image for the CNN.

    Args:
        image_data: RGBA/RGB/gray array from the canvas.
        invert: set True if the drawing is dark ink on a light background.

    Returns:
        (tensor, preview) where tensor has shape (1, 28, 28, 1), float32 in
        [0, 1], and preview is the 28x28 uint8 image the model sees.
        Returns (None, None) if the canvas is empty.
    """
    gray = to_grayscale(image_data)
    if invert:
        gray = 255.0 - gray

    mask = gray > THRESHOLD
    if mask.sum() < MIN_PIXELS:
        return None, None

    # 1. Crop to the bounding box of the stroke.
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    crop = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 2. Resize so the longer side is 20 px, keeping aspect ratio.
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # 3. Paste into the middle of a 28x28 black image.
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.asarray(img, dtype=np.float32)

    # 4. Shift so the center of mass is at the image center (like MNIST).
    cy, cx = ndimage.center_of_mass(canvas)
    shift = (13.5 - cy, 13.5 - cx)
    canvas = ndimage.shift(canvas, shift, order=1, mode="constant", cval=0.0)

    canvas = np.clip(canvas, 0, 255)
    preview = canvas.astype(np.uint8)
    tensor = (canvas / 255.0).astype(np.float32).reshape(1, 28, 28, 1)
    return tensor, preview
