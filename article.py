import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- SRCNN-INSPIRED MODEL ---------
class SimpleSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# --------- UTILS ---------
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def load_image(path):
    img = Image.open(path).convert('L')
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    return tensor

def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(path)

def prepare_training_pair(image, scale):
    # Downsample + Upsample
    low_res = F.interpolate(image, scale_factor=1/scale, mode='bicubic', align_corners=False)
    upsampled = F.interpolate(low_res, size=image.shape[-2:], mode='bicubic', align_corners=False)
    return upsampled, image

# --------- TRAINING FUNCTION ---------
def train_model(train_img, model, scale, epochs=500, lr=1e-4):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        input_img, target_img = prepare_training_pair(train_img, scale)
        output = model(input_img)
        loss = loss_fn(output, target_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[x{scale}] Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

# --------- VALIDATION ---------
def validate(model, val_img, scale):
    model.eval()
    with torch.no_grad():
        low_res = F.interpolate(val_img, scale_factor=1/scale, mode='bicubic', align_corners=False)
        upsampled = F.interpolate(low_res, size=val_img.shape[-2:], mode='bicubic', align_corners=False)
        output = model(upsampled)
        score = psnr(output, val_img)
        save_image(output, f"outputs/sr_x{scale}.png")
        return score

# --------- MAIN PIPELINE ---------
def main():
    os.makedirs("outputs", exist_ok=True)

    train_img = load_image("data/train.jpg")
    val_img = load_image("data/validation.jpg")

    scales = [2, 4, 8, 16, 32]
    for scale in scales:
        print(f"\n--- Training for ×{scale} ---")
        model = SimpleSRNet().to(DEVICE)
        train_model(train_img, model, scale=scale, epochs=500)

        print(f"Validating for ×{scale}...")
        score = validate(model, val_img, scale=scale)
        print(f"PSNR ×{scale}: {score:.2f} dB")

if __name__ == "__main__":
    main()
