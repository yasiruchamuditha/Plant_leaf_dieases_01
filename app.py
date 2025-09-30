import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# ==============================================================
# 1. Load Class Indices
# ==============================================================
def load_class_indices():
    try:
        with open('models/class_indices.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None, "Error: Class indices file not found. Please run the training notebook first."

# ==============================================================
# 2. Load Model
# ==============================================================
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path), None
    except Exception as e:
        return None, f"Error loading model: {e}"

# ==============================================================
# 3. Preprocess Image
# ==============================================================
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================================================
# 4. Predict Function
# ==============================================================
def predict(image, selected_model):
    class_indices, error = load_class_indices()
    if error:
        return error, None, None, None
    
    class_names = {v: k for k, v in class_indices.items()}
    
    model_options = {
        "CNN Model": "models/cnn_model.h5",
        "MobileNetV2": "models/mobilenet_model.h5",
        "ResNet50": "models/resnet_model.h5"
    }
    
    model_path = model_options[selected_model]
    model, error = load_model(model_path)
    if error:
        return error, None, None, None
    
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = class_names[predicted_class_idx]
    
    # Top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_results = [
        f"{i+1}. {class_names[idx]} - {predictions[0][idx] * 100:.2f}%"
        for i, idx in enumerate(top_3_idx)
    ]
    
    return (f"**Prediction:** {predicted_class}\n\n**Confidence:** {confidence:.2f}%",
            "\n".join(top_3_results), image, selected_model)

# ==============================================================
# 5. Compare Models Function
# ==============================================================
def compare_models(image, model_1, model_2, model_3=None):
    class_indices, error = load_class_indices()
    if error:
        return error
    
    class_names = {v: k for k, v in class_indices.items()}
    model_options = {
        "CNN Model": "models/cnn_model.h5",
        "MobileNetV2": "models/mobilenet_model.h5",
        "ResNet50": "models/resnet_model.h5"
    }
    
    processed_img = preprocess_image(image)
    results = []
    
    models_to_compare = [model_1, model_2] if model_3 is None else [model_1, model_2, model_3]
    
    for model_name in models_to_compare:
        model_path = model_options[model_name]
        model, error = load_model(model_path)
        if error:
            results.append(f"{model_name}: {error}")
            continue
        
        preds = model.predict(processed_img)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx] * 100
        pred_class = class_names[pred_idx]
        results.append(f"**{model_name}**\nPrediction: {pred_class}\nConfidence: {confidence:.2f}%")
    
    return "\n\n".join(results)

# ==============================================================
# 6. About Section
# ==============================================================
about_text = """
# 🌱 Plant Disease Detection App

This app uses deep learning models to detect diseases in plant leaves. The models were trained on the **PlantVillage dataset** containing images of healthy and diseased plant leaves.

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
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

**Supported Plants:**
Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
"""

# ==============================================================
# 7. Gradio Interface
# ==============================================================
with gr.Blocks(title="Plant Disease Detection") as demo:
    gr.Markdown("# 🌱 Plant Disease Detection App")
    gr.Markdown("Upload a plant leaf image to detect diseases using deep learning models.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Plant Leaf Image")
            model_select = gr.Dropdown(
                choices=["CNN Model", "MobileNetV2", "ResNet50"],
                label="Select Model",
                value="CNN Model"
            )
            predict_button = gr.Button("Predict")
        
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction Result")
            top_3_output = gr.Textbox(label="Top 3 Predictions")
            uploaded_image = gr.Image(label="Uploaded Image")
    
    with gr.Accordion("Compare Models", open=False):
        gr.Markdown("Compare predictions across different models.")
        comparison_mode = gr.Radio(
            choices=["Model 1 vs Model 2", "Model 1 vs Model 2 vs Model 3"],
            label="Comparison Mode",
            value="Model 1 vs Model 2"
        )
        model_1_select = gr.Dropdown(
            choices=["CNN Model", "MobileNetV2", "ResNet50"],
            label="Model 1",
            value="CNN Model"
        )
        model_2_select = gr.Dropdown(
            choices=["CNN Model", "MobileNetV2", "ResNet50"],
            label="Model 2",
            value="MobileNetV2"
        )
        model_3_select = gr.Dropdown(
            choices=["CNN Model", "MobileNetV2", "ResNet50", None],
            label="Model 3 (optional)",
            value=None
        )
        compare_button = gr.Button("Compare Models")
        comparison_output = gr.Textbox(label="Comparison Results")
    
    with gr.Accordion("About", open=False):
        gr.Markdown(about_text)
    
    predict_button.click(
        fn=predict,
        inputs=[image_input, model_select],
        outputs=[prediction_output, top_3_output, uploaded_image, gr.State()]
    )
    
    compare_button.click(
        fn=compare_models,
        inputs=[image_input, model_1_select, model_2_select, model_3_select],
        outputs=comparison_output
    )

# Launch the app
demo.launch()