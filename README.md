# Dual-Branch CNN-BiLSTM with Attention for Audio Classification

This repository contains the core PyTorch architecture and research report for a custom Indian Classical Music (Raga) classification system.

The model architecture achieved **93.27% Test Accuracy** across 10 classes. This repository features the production-ready **PyTorch** implementation, optimized for edge inference and ONNX export.

## Dataset
The audio samples used to train and evaluate this model are sourced from the [Thaat and Raga Forest (TRF) Dataset](https://www.kaggle.com/datasets/suryamajumder/thaat-and-raga-forest-trf-dataset) on Kaggle.

## Architecture Highlights

- **Custom Audio Processing**: Designed to process high-dimensional audio features including Mel-Spectrograms, Chromagrams, and MFCCs.
- **Dual-Branch Spatial Extraction**: Utilizes parallel 2D Convolutional blocks to extract spatial frequency features without collapsing the temporal dimension.
- **Temporal Modeling**: A Bidirectional LSTM (BiLSTM) processes the sequential feature maps to understand the progression of musical notes (Arohana and Avarohana).
- **Custom Attention Mechanism**: Implements a custom Softmax-Tanh temporal attention layer to dynamically weight the most informative time frames in the audio sequence.
- **Production Optimization**: Includes `export_to_onnx` with dynamic axes for seamless TensorRT deployment, directly addressing sub-100ms real-time latency constraints.

## Files
- `src/`: Contains the core modeling (`model.py`), training (`train.py`), inference script (`inference.py`), and the deployment API (`app.py`).
- `notebooks/`: Contains the Jupyter notebook analysis.
- `models/`: Destination folder for the `.pth` model artifacts.
- `data/`: Contains the audio datasets and features.
