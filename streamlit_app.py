
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Plant Disease Detection App")
st.markdown("Upload an image of a plant leaf to detect diseases using deep learning models.")

# Load class indices
@st.cache_data
def load_class_indices():
    try:
        with open('models/class_indices.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Class indices file not found. Please run the training notebook first.")
        return None

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Main app
def main():
    class_indices = load_class_indices()
    if class_indices is None:
        return

    # Create reverse mapping
    class_names = {v: k for k, v in class_indices.items()}

    # Model selection
    st.sidebar.header("Model Selection")
    model_options = {
        "CNN Model": "models/cnn_model.h5",
        "MobileNetV2": "models/mobilenet_model.h5", 
        "ResNet50": "models/resnet_model.h5"
    }

    selected_model = st.sidebar.selectbox("Choose a model:", list(model_options.keys()))
    model_path = model_options[selected_model]

    # Load selected model
    model = load_model(model_path)
    if model is None:
        st.error(f"Could not load {selected_model}")
        return

    st.sidebar.success(f"✅ {selected_model} loaded successfully!")

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf for disease detection"
    )

    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.subheader("🔍 Prediction Results")

            # Make prediction
            with st.spinner("Analyzing image..."):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class_idx] * 100
                predicted_class = class_names[predicted_class_idx]

            # Display results
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Show top 3 predictions
            st.subheader("📊 Top 3 Predictions")
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]

            for i, idx in enumerate(top_3_idx):
                class_name = class_names[idx]
                prob = predictions[0][idx] * 100
                st.write(f"{i+1}. **{class_name}** - {prob:.2f}%")
                st.progress(prob/100)

    # Information section
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This app uses deep learning models to detect diseases in plant leaves. The models were trained on the PlantVillage dataset containing images of healthy and diseased plant leaves.

        **Available Models:**
        - **CNN Model**: Custom convolutional neural network with L2 regularization
        - **MobileNetV2**: Lightweight model optimized for mobile deployment
        - **ResNet50**: Deep residual network with skip connections

        **Supported Plant Diseases:**
        - Pepper Bell Bacterial Spot
        - Pepper Bell Healthy
        - Potato Early Blight
        - Potato Late Blight
        - Potato Healthy
        - Tomato Bacterial Spot
        - Tomato Early Blight
        - Tomato Late Blight
        - Tomato Leaf Mold
        - Tomato Septoria Leaf Spot
        - Tomato Spider Mites
        - Tomato Target Spot
        - Tomato Yellow Leaf Curl Virus
        - Tomato Mosaic Virus
        - Tomato Healthy
        """)

if __name__ == "__main__":
    main()
