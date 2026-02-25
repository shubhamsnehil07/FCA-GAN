import torch
import math

def get_dct_weights(height, width, channel, num_freq=16):
    indices = []
    for u in range(height):
        for v in range(width):
            indices.append((u, v))
    indices.sort(key=lambda x: x[0] + x[1])
    indices = indices[:num_freq]

    dct_weights = torch.zeros(channel, num_freq, height, width)
    for i, (u, v) in enumerate(indices):
        for h in range(height):
            for w in range(width):
                weight = math.cos(math.pi * u * (h + 0.5) / height) * \
                         math.cos(math.pi * v * (w + 0.5) / width)
                if u == 0: weight *= 1.0 / math.sqrt(2)
                if v == 0: weight *= 1.0 / math.sqrt(2)
                dct_weights[:, i, h, w] = weight * (2.0 / math.sqrt(height * width))
    return dct_weights
