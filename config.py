import torch
import os
from datetime import datetime

CURRENT_USER = "shubhamsnehil07"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
TRAIN_ROOT = r'C:\Users\coco2017\test2017'
VAL_ROOT   = r'C:\Users\coco2017\val2017'
OUTPUT_BASE = r'C:\Users\output_fca'

# Training Hyperparameters
TRAIN_SAMPLES = 3000   
VAL_SAMPLES   = 600   
SEED = 42
BATCH_SIZE = 24
EPOCHS = 200
LAMBDA_L1 = 50
LR = 2e-4
SAVE_EVERY = 5
