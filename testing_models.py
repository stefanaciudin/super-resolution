import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import load_image, downsample, show_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1 / mse.item())
# ----------------------------
# MODELUL - aceeași arhitectură
# ----------------------------

class EfficientSRCNN(nn.Module):
    def __init__(self):
        super(EfficientSRCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)

# ----------------------------
# ÎNCĂRCARE MODEL
# ----------------------------

model = EfficientSRCNN().to(device)
checkpoint = torch.load("models/model_x2.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model încărcat (epoca {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")


print("\nPSNR pe imaginea de validare:")
for scale in [2, 4, 8, 16, 32]:
    test_path = f"data/validation.jpg"
    test_img = load_image(test_path)
    down, up = downsample(test_img, scale)
    with torch.no_grad():
        output = model(up)
        score = psnr(output, test_img)
        print(f"scale: {scale}-> PSNR: {score:.2f} dB")
        show_images(test_img, output, f"Super-rezoluție x{scale}")


