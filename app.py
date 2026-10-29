"""Food Freshness Detection - Streamlit app.

Classifies a fruit photo as Fresh or Rotten with a MobileNetV2 model trained in
notebooks/Food_Freshness_Detection.ipynb.
Run locally:  streamlit run app.py
"""
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="Food Freshness Detection", page_icon="🍎", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "food_freshness_model.keras"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Rotten"]      # label 0 = Fresh, 1 = Rotten (same as the notebook)
LOW_CONFIDENCE = 0.70


# ----------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading model (first load can take a moment)...")
def load_model_from_path(path: str):
    from tensorflow import keras           # imported here so the page appears quickly
    return keras.models.load_model(path)


@st.cache_resource(show_spinner="Loading uploaded model...")
def load_model_from_bytes(data: bytes):
    from tensorflow import keras
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return keras.models.load_model(tmp_path)
    finally:
        os.remove(tmp_path)


def get_model():
    """Use model/food_freshness_model.keras if present; otherwise let the user upload one."""
    if MODEL_PATH.exists():
        try:
            return load_model_from_path(str(MODEL_PATH))
        except Exception as e:
            st.error("The model file exists but could not be loaded. This usually means the "
                     "TensorFlow version in requirements.txt differs from the one used for training.")
            st.code(str(e)[:600])
            return None

    st.warning("No trained model found at `model/food_freshness_model.keras`. "
               "Train it with the notebook, then add the file to the repo. "
               "For a quick test you can upload it below.")
    uploaded = st.file_uploader("Upload food_freshness_model.keras", type=["keras"], key="model_upload")
    if uploaded is None:
        return None
    try:
        return load_model_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error("That file could not be loaded as a model.")
        st.code(str(e)[:600])
        return None


# ----------------------------------------------------------------- image + prediction
def read_image(file):
    """Return an RGB PIL image or raise ValueError with a friendly message."""
    try:
        data = file.getvalue()
        Image.open(io.BytesIO(data)).verify()          # detects corrupted files
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)                           # fix rotated phone photos
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ValueError("This file could not be read as an image. Please upload a valid JPG or PNG.")


def predict(model, img: Image.Image):
    arr = np.asarray(img.resize(IMG_SIZE), dtype="float32")
    arr = np.expand_dims(arr, 0)          # raw 0-255 pixels; scaling to [-1, 1] is inside the model
    p_rotten = float(model.predict(arr, verbose=0)[0][0])
    label = CLASS_NAMES[1] if p_rotten >= 0.5 else CLASS_NAMES[0]
    return label, p_rotten


def show_result(model, file):
    try:
        img = read_image(file)
    except ValueError as e:
        st.error(f"❌ {e}")
        return
    st.image(img, caption="Your image", use_container_width=True)
    with st.spinner("Analysing..."):
        try:
            label, p_rotten = predict(model, img)
        except Exception:
            st.error("❌ Prediction failed for this image. Please try another photo.")
            return

    confidence = max(p_rotten, 1 - p_rotten)
    if label == "Fresh":
        st.success(f"### ✅ Fresh  \nConfidence: **{confidence * 100:.1f}%**")
    else:
        st.error(f"### 🚫 Rotten  \nConfidence: **{confidence * 100:.1f}%**")
    if confidence < LOW_CONFIDENCE:
        st.warning("Low confidence. The photo may be unclear, or the item may not be one of "
                   "the fruits the model was trained on.")

    st.write("**Class probabilities**")
    st.progress(1 - p_rotten, text=f"Fresh: {(1 - p_rotten) * 100:.1f}%")
    st.progress(p_rotten, text=f"Rotten: {p_rotten * 100:.1f}%")


# ----------------------------------------------------------------- page
st.title("🍎 Food Freshness Detection")
st.caption("Deep learning (MobileNetV2 transfer learning) · classifies a fruit photo as **Fresh** or **Rotten**")

with st.sidebar:
    st.header("About")
    st.write("Upload a clear photo of a single fruit. The model was trained on the "
             "*Fresh and Rotten Fruits* dataset (Sultana et al., 2022) with 8 fruit types: "
             "apple, banana, orange, grape, guava, jujube, pomegranate and strawberry.")
    st.header("Limitations")
    st.write("- Other fruits or non-fruit images give unreliable results.\n"
             "- Blurry or dark photos reduce accuracy.\n"
             "- It only says fresh or rotten, not ripeness.")
    if METRICS_PATH.exists():
        try:
            m = json.loads(METRICS_PATH.read_text())
            st.header("Test-set results")
            st.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
            st.write(f"Precision (Rotten): {m['precision_rotten']:.3f}  \n"
                     f"Recall (Rotten): {m['recall_rotten']:.3f}  \n"
                     f"F1 (Rotten): {m['f1_rotten']:.3f}")
        except Exception:
            pass

model = get_model()

if model is not None:
    tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])
    with tab_upload:
        file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])
        if file is not None:
            show_result(model, file)
        else:
            st.info("Upload a JPG or PNG image to get a prediction.")
    with tab_camera:
        shot = st.camera_input("Take a photo of the fruit")
        if shot is not None:
            show_result(model, shot)
