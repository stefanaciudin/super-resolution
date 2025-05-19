import torch
from torch.cuda import device
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from math import log10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * log10(1 / mse.item())

def load_image(path, resize_to=None):
    image = Image.open(path).convert("RGB")
    if resize_to:
        image = image.resize(resize_to, Image.BICUBIC)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

def downsample(img, scale):
    _, _, h, w = img.shape
    down = F.interpolate(img, scale_factor=1/scale, mode='bicubic', align_corners=False)
    up = F.interpolate(down, size=(h, w), mode='bicubic', align_corners=False)
    return down, up

def show_images(original, reconstructed, title):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Original")
    axs[1].imshow(np.clip(reconstructed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), 0, 1))
    axs[1].set_title("Reconstruit")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()