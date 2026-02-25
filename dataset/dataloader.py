import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None, seed=None):
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        self.image_paths = sorted(list(set(self.image_paths)))

        if max_samples and max_samples < len(self.image_paths):
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.image_paths))[:max_samples]
            self.image_paths = [self.image_paths[i] for i in indices]
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        gray = img.mean(dim=0, keepdim=True)
        return gray, img
