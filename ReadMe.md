# VP-SAT: Vanishing Point Spatial Attention Transformer

## Objective
In this project, you will explore and improve the ConVexNet model, a Transformer-based network designed to detect vanishing points (VPs) from image patches. The architecture consists of three key modules:

1. **Feature Extractor (CNN-based)**
2. **Transformer Encoder**
3. **Vanishing Point Prediction Head**

Your goal is to experiment with enhancements to any of these modules, or combinations thereof, to improve vanishing point prediction accuracy.

---

## Getting Started

### Project Structure
```
VPSAT/
├── checkpoint
├── config
│   ├── model_config.yaml
├── data
│   ├── dataset.py
│   └── su3
├── models/
│   ├── config.py.py
|   ├── transformer_encoder.py
│   └── vpsat.py
├── utils/
├── train.py
├── eval.py
└── ReadMe.md
```

---

## Installation

### Requirements
- python=3.9
- matplotlib
- numpy
- PyYAML
- scikit-image
- torch
- torchvision
- tqdm

### Setup
Clone the repository:
```bash
git clone https://github.com/PSU-LPAC/VPSAT
cd VPSAT
```

Install dependencies: It is encouraged to install miniconda3 (or anaconda)
```bash
conda create -n vpsat python=3.9
pip3 install -r requirements.txt
```

---

## Data Preparation

This project uses the **WireframeDataset**. Follow these steps to prepare the dataset:

- Download the Wireframe dataset.
- Extract and place the dataset in the `data/` directory.
- Ensure the dataset directory has `train/`, `valid/`, and `test/` splits.

---

## Training

To start training, run:

```bash
python train.py
```

You can configure training settings in `model_config.yaml`, including:

- Batch size
- Learning rate
- Number of epochs
- Transformer parameters (layers, heads, embedding dimensions)
- GPU device specification

---


**Metrics Used:**
- Cosine similarity (loss)
- Angular accuracy (degrees)

---

## Network Architecture

### 1. Feature Extractor (CNN)
- Uses depthwise-separable convolutions for efficient feature extraction.
- Includes residual connections and global average pooling for compact embeddings.

### 2. Transformer Encoder
- Employs positional encodings and multi-head self-attention.
- Captures global context and structural relations across image patches.

### 3. Vanishing Point Prediction Head
- Attention-based pooling aggregates Transformer outputs.
- Multi-Layer Perceptron (MLP) predicts final VP coordinates.

---

## Improving the Model

You are encouraged to explore improvements in the following areas:

- **Feature Extractor:** Experiment with advanced CNN architectures, convolution types, and normalization techniques.
- **Transformer Encoder:** Test different attention mechanisms, positional encodings, or increase depth and complexity.
- **Prediction Head:** Investigate deeper MLP architectures, alternative pooling strategies, or intermediate VP representations.

Document and justify your modifications thoroughly.

---

## Submission Guidelines

Your final submission should include:

- Clearly documented code.
- A detailed report explaining your baseline results and improvements.
- Quantitative and qualitative analysis demonstrating the effectiveness of your approach.

