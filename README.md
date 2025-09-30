# 🌱 Plant Disease Detection System

A comprehensive deep learning solution for detecting diseases in plant leaves using computer vision and neural networks. This project implements multiple state-of-the-art models to classify plant diseases from leaf images.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project uses deep learning to identify plant diseases from leaf images, helping farmers and agricultural professionals make informed decisions about crop health. The system supports multiple plant types including tomatoes, potatoes, and peppers.

### 🔍 Supported Plant Diseases

- **Pepper Bell**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy  
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

## 🚀 Features

- **Multiple Model Architectures**: CNN, MobileNetV2, EfficientNet, ResNet50
- **High Accuracy**: Achieves 90%+ accuracy on test data
- **Web Interface**: Interactive Streamlit app for easy deployment
- **Real-time Predictions**: Fast inference with confidence scores
- **Robust Error Handling**: Graceful fallbacks for model loading issues
- **Data Augmentation**: Advanced image preprocessing techniques

## 🏗️ Model Architectures

### 1. Custom CNN

- L2 regularization and batch normalization
- Optimized for plant disease classification
- Lightweight and fast inference

### 2. MobileNetV2

- Pre-trained on ImageNet
- Mobile-optimized architecture
- Excellent balance of accuracy and speed

### 3. ResNet50

- Deep residual learning
- Pre-trained weights with fine-tuning
- Robust performance across diverse conditions

## 📊 Dataset

- **Source**: PlantVillage Dataset (Kaggle)
- **Images**: ~54,000+ plant leaf images
- **Classes**: 16 different plant disease categories
- **Split**: 70% Training, 20% Validation, 10% Testing
- **Resolution**: 224x224 pixels (standardized)

## 🛠️ Installation

### Prerequisites

```bash
python >= 3.8
pip >= 21.0
```

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yasiruchamuditha/Plant_leaf_dieases_01.git
cd Plant_leaf_dieases_01

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
tensorflow>=2.15.0
streamlit>=1.28.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
pillow>=8.0.0
kagglehub>=0.3.0
```

## 🎮 Usage

### 1. Training Models

Run the Jupyter notebook to train all models:

```bash
jupyter notebook train.ipynb
```

Execute all cells in sequence to:

- Download and prepare the dataset
- Train multiple model architectures
- Evaluate performance on test data
- Save trained models

### 2. Web Application

Launch the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

The app provides:

- Model selection dropdown
- Image upload functionality
- Real-time disease prediction
- Confidence scores and top-3 predictions
- Interactive user interface
### This is only for locally testing purposes only.Application is hosted with hugging face for production

### 3. Command Line Prediction

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('models/resnet50_model.h5')

# Preprocess image
image = Image.open('path/to/leaf_image.jpg').resize((224, 224))
img_array = np.array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction) * 100

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

## 📁 Project Structure

```text
Plant_leaf_dieases_01/
├── train.ipynb              # Main training notebook
├── app.py           # Web application
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
├── .gitignore                # Git ignore rules
├── data/                     # Dataset (auto-generated)
│   ├── train/               # Training images
│   ├── val/                 # Validation images
│   └── test/                # Test images
├── models/                   # Trained models (auto-generated)
│   ├── cnn_model.h5
│   ├── mobilenet_model.h5
│   ├── resnet50_model.h5
│   └── class_indices.json
└── datasets/                 # Raw dataset cache
    └── emmarex/
```

## 🎯 Model Performance

| Model | Test Accuracy | Parameters | Inference Time |
|-------|--------------|------------|---------------|
| Custom CNN | 81.3% | 2.1M | 15ms |
| MobileNetV2 | 93.38% | 3.5M | 12ms |
| ResNet50 | 94.7% | 25.6M | 25ms |

*EfficientNet performance with random weights (fallback mode)

## 🔧 Troubleshooting

### Common Issues

#### 1. EfficientNet Shape Mismatch Error

```text
ValueError: Shape mismatch in layer #1 (named stem_conv)
```

**Solution**: The notebook automatically handles this with a fallback to random weights.

#### 2. Out of Memory Error

```text
ResourceExhaustedError: OOM when allocating tensor
```

**Solution**: Reduce batch size in the notebook or use a machine with more RAM/GPU memory.

#### 3. Dataset Download Issues

```text
ConnectionError: Failed to download dataset
```

**Solution**: Check internet connection and Kaggle API credentials.

### Performance Optimization

- **GPU Training**: Use CUDA-enabled TensorFlow for faster training
- **Mixed Precision**: Enable mixed precision training for better performance
- **Model Quantization**: Convert models to TensorFlow Lite for mobile deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

### Yasiru Chamuditha

- GitHub: [@yasiruchamuditha](https://github.com/yasiruchamuditha)
- Project Link: [https://github.com/yasiruchamuditha/Plant_leaf_dieases_01](https://github.com/yasiruchamuditha/Plant_leaf_dieases_01)

---

⭐ **Star this repository if you find it helpful!**

## 🔮 Future Enhancements

- [ ] Mobile app development (iOS/Android)
- [ ] Integration with agricultural IoT devices
- [ ] Real-time disease progression tracking
- [ ] Multi-language support for global adoption
- [ ] Integration with agricultural databases
- [ ] Drone-based image capture support
- [ ] Treatment recommendation system
- [ ] Weather correlation analysis
