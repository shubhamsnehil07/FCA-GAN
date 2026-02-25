import torch
import cv2
from models.generator import Generator
import config


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = img[:, :, 0:1] / 50. - 1.
    L = torch.tensor(L).permute(2, 0, 1).unsqueeze(0).float()
    return L


def inference(image_path, model_path):
    device = config.DEVICE
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    L = load_image(image_path).to(device)
    with torch.no_grad():
        fake_ab = model(L)

    print("Colorization complete.")


if __name__ == "__main__":
    inference("input.jpg", "generator.pth")
