# Online Hand Gesture Recognition

## Overview
This project focuses on developing an online hand gesture recognition system. Utilizing advanced deep learning techniques, it aims to accurately recognize and interpret hand gestures in real-time. The system employs a sophisticated CNN-Transformer architecture to process and analyze video data, making it capable of understanding complex hand gestures.

## Features
- **Real-Time Gesture Recognition**: Detects and interprets hand gestures in real-time.
- **Advanced CNN-Transformer Model**: Leverages the strengths of both CNNs for spatial feature extraction and Transformers for capturing temporal dynamics.
- **Variable-Length Sequence Handling**: Accommodates gestures of varying durations with effective padding and masking strategies.
- **High Accuracy and Efficiency**: Optimized for both high accuracy in gesture recognition and operational efficiency in online settings.

## Requirements
- Python 3.8 or above
- PyTorch
- Torchvision
- CUDA (for GPU acceleration)
- Other dependencies listed in `requirements.txt`

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/KenanKhauto/online-gesture-recognition.git
cd online-gesture-recognition
pip install -r requirements.txt
