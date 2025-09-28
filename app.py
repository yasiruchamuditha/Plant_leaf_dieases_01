import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# =========================
# Load Class Names Safely
# =========================
MODEL_PATH = "models/efficientnet_model.h5"  # or whichever you want by default
CLASS_JSON_PATH = "models/class_indices.json"

if os.path.exists(CLASS_JSON_PATH):
    with open(CLASS_JSON_PATH, "r") as f:
        class_indices = json.load(f)
    # Ensure order by index
    class_names = [cls for cls, idx in sorted(class_indices.items(), key=lambda x: x[1])]
else:
    st.error("❌ class_indices.json not found! Please save it after training.")
    st.stop()

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

# Sanity check: model outputs vs labels
output_neurons = model.output_shape[-1]
if len(class_names) != output_neurons:
    st.error(f"❌ Label mismatch: model outputs {output_neurons} classes but class_indices.json has {len(class_names)}.")
    st.stop()

st.success(f"Loaded model with {len(class_names)} classes ✅")

# =========================
# Streamlit UI
# =========================
st.title("🌱 Plant Disease Detection")
st.write("Upload a plant leaf image to predict if it's healthy or diseased.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    confidence = float(np.max(preds))

    # Show Results
    st.subheader(f"Prediction: {pred_class}")
    st.write(f"Confidence: {confidence*100:.2f}%")

    # Confidence Chart
    st.bar_chart(preds[0])
