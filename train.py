import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from config import *
from models.generator import GeneratorUNet
from models.discriminator import Discriminator
from dataset.dataloader import ColorizationDataset

def train():
    # Setup Dirs
    run_dir = os.path.join(OUTPUT_BASE, f"run_{CURRENT_TIMESTAMP}")
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)

    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_ds = ColorizationDataset(TRAIN_ROOT, transform, max_samples=TRAIN_SAMPLES, seed=SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    gen = GeneratorUNet().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    
    opt_G = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(EPOCHS):
        for i, (gray, color) in enumerate(train_loader):
            gray, color = gray.to(DEVICE), color.to(DEVICE)
            color_tanh = color * 2 - 1

            # Train D
            opt_D.zero_grad()
            fake = gen(gray).detach()
            loss_D = 0.5 * (criterion_GAN(disc(gray, color_tanh), torch.ones_like(disc(gray, color_tanh))) + 
                           criterion_GAN(disc(gray, fake), torch.zeros_like(disc(gray, fake))))
            loss_D.backward(); opt_D.step()

            # Train G
            opt_G.zero_grad()
            fake = gen(gray)
            loss_G = criterion_GAN(disc(gray, fake), torch.ones_like(disc(gray, fake))) + \
                     criterion_L1(fake, color_tanh) * LAMBDA_L1
            loss_G.backward(); opt_G.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i} Loss D: {loss_D.item():.4f} G: {loss_G.item():.4f}")

        # Save Sample
        save_image((fake[:4] + 1) / 2, os.path.join(run_dir, 'samples', f"epoch_{epoch}.png"))

if __name__ == "__main__":
    train()
