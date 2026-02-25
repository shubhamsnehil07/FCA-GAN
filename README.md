FCA-GAN: Frequency Cue Attention GAN for Image Colorization
Overview

FCA-GAN is a GAN-based image colorization framework that integrates:

Spatial Attention Mechanism

Frequency-Domain Cues (FFT-based features)

Multi-loss optimization for realistic color synthesis

The model leverages both spatial and frequency information to improve texture consistency and fine detail restoration during colorization.

Key Contributions

Attention-guided generator for focused feature learning

Frequency-aware feature fusion using FFT

Adversarial + L1 + Perceptual + Frequency consistency loss

PatchGAN discriminator

Modular PyTorch implementation

Architecture

Generator:

U-Net backbone

Frequency feature extraction block

Spatial attention modules

Feature fusion layer

Discriminator:

PatchGAN

Multi-scale feature discrimination

Why Frequency Cues?

Standard GAN colorization works in spatial domain only.

FCA-GAN introduces:

FFT magnitude features

High-frequency texture preservation

Edge-aware reconstruction

This improves:

Texture sharpness

Structural consistency

Fine-detail coloring

Training Objective

Total Loss =

Adversarial Loss

L1 Reconstruction Loss

Perceptual Loss

Frequency Consistency Loss

Installation
git clone https://github.com/yourusername/FCA-GAN.git
cd FCA-GAN
pip install -r requirements.txt
Train
python train.py
Inference
python inference.py --image path/to/grayscale.jpg
Tech Stack

Python

PyTorch

OpenCV

NumPy

Matplotlib

Future Work

Multi-scale frequency learning

Transformer-based attention

Diffusion refinement stage

FID and LPIPS evaluation

Author

Shubham
MSc Computer Science — NIT Trichy
