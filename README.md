# FCA-GAN: Frequency Cue Attention GAN for Image Colorization

## Overview

FCA-GAN is a GAN-based image colorization framework that integrates:

- Spatial Attention Mechanism
- Frequency-Domain Cues (FFT-based features)
- Multi-loss optimization for realistic color synthesis

The model leverages both spatial and frequency information to improve texture consistency and fine detail restoration during colorization.

---

## Project Structure
```
FCA-GAN/
│
├── models/
│   ├── generator.py
│   ├── discriminator.py
│   ├── attention.py
│   ├── frequency.py
│
├── losses/
│   ├── adversarial.py
│   ├── perceptual.py
│   ├── frequency_loss.py
│
├── dataset/
│   ├── dataloader.py
│
├── train.py
├── inference.py
├── config.py
└── requirements.txt
```
## Key Features

- Attention-guided generator architecture
- Frequency-aware learning using FFT
- Adversarial + L1 + Perceptual + Frequency Consistency loss
- PatchGAN discriminator
- Modular PyTorch implementation

---

## Architecture

### Generator
- U-Net backbone
- Frequency feature extraction block
- Spatial attention modules
- Feature fusion layer

### Discriminator
- PatchGAN architecture
- Multi-scale discrimination

---

## Why Frequency Cues?

Traditional GAN colorization works only in the spatial domain.

FCA-GAN introduces:
- FFT magnitude features
- High-frequency texture preservation
- Edge-aware reconstruction

This improves:
- Texture sharpness
- Structural consistency
- Fine-detail coloring

---

## Loss Function

Total Loss =

- Adversarial Loss
- L1 Reconstruction Loss
- Perceptual Loss
- Frequency Consistency Loss

---

## Installation

Clone the repository:

git clone https://github.com/shubhamsnehil07/FCA-GAN.git

cd FCA-GAN

Install dependencies:

```pip install -r requirements.txt```

---

## Training

```python train.py```

---

## Inference

```python inference.py --image path/to/grayscale.jpg```

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib

---

## Future Improvements

- Multi-scale frequency learning
- Transformer-based attention
- Diffusion-based refinement
- FID evaluation metrics

---

## Author

Shubham Snehil  
MSc Computer Science  
National Institute of Technology, Trichy
