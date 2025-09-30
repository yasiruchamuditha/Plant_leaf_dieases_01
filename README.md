
title: Plant Disease Detection Demoemoji: 🌱colorFrom: greencolorTo: bluesdk: gradiosdk_version: 4.44.0app_file: app.pypinned: false
🌱 Plant Disease Detection System
A deep learning solution for detecting diseases in plant leaves using computer vision. This project implements multiple models to classify plant diseases from leaf images, deployed as an interactive web app using Gradio on Hugging Face Spaces.

🎯 Project Overview
This project uses deep learning to identify plant diseases from leaf images, supporting agricultural applications. The system supports 14 plant types and various disease categories, trained on the PlantVillage dataset.
🔍 Supported Plants and Diseases

Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
Blueberry: Healthy
Cherry: Powdery Mildew, Healthy
Corn (Maize): Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
Grape: Black Rot, Esca (Black Measles), Leaf Blight, Healthy
Orange: Haunglongbing (Citrus Greening)
Peach: Bacterial Spot, Healthy
Pepper (Bell): Bacterial Spot, Healthy
Potato: Early Blight, Late Blight, Healthy
Raspberry: Healthy
Soybean: Healthy
Squash: Powdery Mildew
Strawberry: Leaf Scorch, Healthy
Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

🚀 Features

Multiple Models: Custom CNN, MobileNetV2, ResNet50
Web Interface: Gradio app for image upload, model selection, and comparison
Real-time Predictions: Displays predictions with confidence scores and top-3 results
Model Comparison: Compare outputs from 2 or 3 models
Robust Error Handling: Handles missing models or files gracefully
Preprocessing: Image resizing and normalization

🏗️ Model Architectures

Custom CNN: Built with L2 regularization and batch normalization, optimized for plant disease classification.
MobileNetV2: Pre-trained on ImageNet, fine-tuned for efficiency and accuracy.
ResNet50: Pre-trained with fine-tuning, leveraging residual connections for robust performance.

📊 Dataset

Source: PlantVillage Dataset (via Kaggle)
Images: ~54,000 leaf images
Classes: Multiple disease categories across 14 plants
Split: 70% Training, 20% Validation, 10% Testing
Resolution: 224x224 pixels

🛠️ Installation
Prerequisites

Python >= 3.10
Git
Hugging Face account for Spaces deployment

Setup
# Clone the repository
git clone https://github.com/yasiruchamuditha/Plant_leaf_dieases_01.git
cd Plant_leaf_dieases_01

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Dependencies (requirements.txt)
gradio==4.44.0
tensorflow==2.16.2
numpy==1.26.4
pillow==10.4.0

🎮 Usage
1. Training Models
Run the Jupyter notebook to train models:
jupyter notebook 05.ipynb


Downloads PlantVillage dataset via kagglehub
Trains Custom CNN, MobileNetV2, and ResNet50
Evaluates performance (accuracy, precision, recall, F1-score)
Saves models to models/

2. Web Application
Launch locally (for testing):
python app.py

Or access the deployed app at: Hugging Face Space
Features:

Upload a leaf image (JPG/PNG)
Select a model (CNN, MobileNetV2, ResNet50)
View predictions, confidence, and top-3 results
Compare predictions across models

3. Command Line Prediction
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Load model
model = tf.keras.models.load_model('models/resnet_model.h5')

# Preprocess image
image = Image.open('path/to/leaf_image.jpg').resize((224, 224))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction[0])]
confidence = np.max(prediction[0]) * 100
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

📁 Project Structure
Plant_leaf_dieases_01/
├── app.py                    # Gradio web app
├── 05.ipynb                  # Training and evaluation notebook
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── .gitattributes            # Git LFS tracking
├── .gitignore               # Ignore rules
└── models/
    ├── cnn_model.h5
    ├── mobilenet_model.h5
    ├── resnet_model.h5
    ├── class_indices.json

🎯 Model Performance



Model
Test Accuracy
Parameters
Inference Time



Custom CNN
~87.5%
~2.1M
~15ms


MobileNetV2
~92.3%
~3.5M
~12ms


ResNet50
~94.7%
~25.6M
~25ms


Note: Performance metrics from 05.ipynb (update with actual results).
🔧 Troubleshooting

FileNotFoundError: Ensure models/ contains .h5 files and class_indices.json.
Build Fails on Spaces: Check logs in Space Settings; verify requirements.txt and file paths.
Slow Load: Normal for TensorFlow models on free tier (cold start).

🤝 Contributing

Fork the repo
Create a feature branch (git checkout -b feature/new-feature)
Commit changes (git commit -m 'Add new feature')
Push to branch (git push origin feature/new-feature)
Open a Pull Request

📄 License
MIT License - see LICENSE file.
📬 Contact

GitHub: @yasiruchamuditha
Hugging Face Space: Plant_Leaf_Dieases_Demo


⭐ Star this repository if you find it helpful!