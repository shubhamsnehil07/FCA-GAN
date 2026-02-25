import os
import cv2
import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        L = img[:, :, 0:1] / 50. - 1.
        ab = img[:, :, 1:] / 128.

        L = torch.tensor(L).permute(2, 0, 1).float()
        ab = torch.tensor(ab).permute(2, 0, 1).float()

        return L, ab
