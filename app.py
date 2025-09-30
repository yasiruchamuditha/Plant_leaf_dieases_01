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
st.markdown("")

st.markdown("""        
        **Supported Plants:**
        Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
        """)


# ==============================================================
# 1. Load Class Indices
# ==============================================================
@st.cache_data
def load_class_indices():
    try:
        with open('models/class_indices.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Class indices file not found. Please run the training notebook first.")
        return None

# ==============================================================
# 2. Load Model
# ==============================================================
@st.cache_resource
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==============================================================
# 3. Preprocess Image
# ==============================================================
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================================================
# 4. Main App
# ==============================================================
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
                
                # Validate predictions
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    st.error("Model predictions contain NaN or infinite values.")
                    return
                
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx]) * 100  # Convert to percentage
                predicted_class = class_names[predicted_class_idx]

            # Display results
            st.success(f"**Prediction:** {predicted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            # Show top 3 predictions
            st.subheader("📊 Top 3 Predictions")
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]

            for i, idx in enumerate(top_3_idx):
                class_name = class_names[idx]
                prob = float(predictions[0][idx]) * 100  # Convert to percentage
                st.write(f"{i+1}. **{class_name}** - {prob:.2f}%")
                progress_value = float(prob / 100)  # Convert to 0-1 range for st.progress
                if 0 <= progress_value <= 1:
                    st.progress(progress_value)
                else:
                    st.warning(f"Invalid progress value for {class_name}: {progress_value}")



    # ==============================================================
    # 5. Model Comparison Section
    # ==============================================================
    with st.expander("⚖️ Model Comparison"):
        st.markdown("""
        Compare predictions across different models to see which performs better 
        on your uploaded image.
        """)

        comparison_mode = st.radio(
            "Choose comparison mode:",
            ("Model 1 vs Model 2", "Model 1 vs Model 2 vs Model 3")
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            processed_img = preprocess_image(image)

            if comparison_mode == "Model 1 vs Model 2":
                selected_models = st.multiselect(
                    "Pick two models to compare:",
                    list(model_options.keys()),
                    default=["CNN Model", "MobileNetV2"]
                )

                if len(selected_models) == 2:
                    col1, col2 = st.columns(2)
                    for i, model_name in enumerate(selected_models):
                        model_path = model_options[model_name]
                        model_cmp = load_model(model_path)
                        preds = model_cmp.predict(processed_img)
                        
                        # Validate predictions
                        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                            st.error(f"Model {model_name} predictions contain NaN or infinite values.")
                            continue
                        
                        pred_idx = np.argmax(preds[0])
                        confidence = float(preds[0][pred_idx]) * 100
                        pred_class = class_names[pred_idx]

                        with [col1, col2][i]:
                            st.subheader(model_name)
                            st.write(f"**Prediction:** {pred_class}")
                            st.write(f"**Confidence:** {confidence:.2f}%")

            elif comparison_mode == "Model 1 vs Model 2 vs Model 3":
                cols = st.columns(3)
                for i, (model_name, model_path) in enumerate(model_options.items()):
                    model_cmp = load_model(model_path)
                    preds = model_cmp.predict(processed_img)
                    
                    # Validate predictions
                    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                        st.error(f"Model {model_name} predictions contain NaN or infinite values.")
                        continue
                    
                    pred_idx = np.argmax(preds[0])
                    confidence = float(preds[0][pred_idx]) * 100
                    pred_class = class_names[pred_idx]

                    with cols[i]:
                        st.subheader(model_name)
                        st.write(f"**Prediction:** {pred_class}")
                        st.write(f"**Confidence:** {confidence:.2f}%")

    # ==============================================================
    # 6. About Section
    # ==============================================================
    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This app uses deep learning models to detect diseases in plant leaves.  
        The models were trained on the **PlantVillage dataset** containing images of healthy and diseased plant leaves.

        **Available Models:**
        - **CNN Model**: Custom convolutional neural network with L2 regularization
        - **MobileNetV2**: Lightweight model optimized for mobile deployment
        - **ResNet50**: Deep residual network with skip connections

        **Supported Plant Diseases:**
        - Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy  
        - Blueberry: Healthy  
        - Cherry (including sour): Powdery Mildew, Healthy  
        - Corn (Maize): Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy  
        - Grape: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy  
        - Orange: Haunglongbing (Citrus Greening)  
        - Peach: Bacterial Spot, Healthy  
        - Pepper (Bell): Bacterial Spot, Healthy  
        - Potato: Early Blight, Late Blight, Healthy  
        - Raspberry: Healthy  
        - Soybean: Healthy  
        - Squash: Powdery Mildew  
        - Strawberry: Leaf Scorch, Healthy  
        - Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot,  
          Spider Mites (Two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy  
                    
        **Supported Plants:**
        Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach,  
        Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
        """)

# ==============================================================
# 7. Run App
# ==============================================================
if __name__ == "__main__":
    main()