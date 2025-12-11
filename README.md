# Brain Tumor Classification

Deep learning model for classifying brain tumors from MRI images using PyTorch.

**ELEC576 Final Project - Fall 2025**  
Alex Smith • Sugiarto Wibowo • Tongda Yin

## Overview

This project implements and compares deep learning models (MobileNetV3 and VGG16) for multi-class brain tumor classification from MRI scans.

## Setup

1. **Install dependencies:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch torchvision numpy matplotlib pillow scikit-learn torchinfo tensorboard jupyter
```

2. **Run the notebook:**
```bash
jupyter notebook Final_Project_Brain_Tumors_Detection.ipynb
```

The dataset will be automatically downloaded from GitHub on first run.

## Model Performance

- **MobileNetV3-Large**: 82.23% test accuracy, ~15MB model size, fast inference
- **VGG16**: Comparison model with feature extraction frozen

## Features

- Data augmentation for improved generalization
- Early stopping and learning rate scheduling
- TensorBoard logging
- Inference time benchmarking
- Model size comparison

## Requirements

- Python 3.12+
- PyTorch 2.9+
- CUDA-capable GPU (optional, for faster training)
