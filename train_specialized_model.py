import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.transforms import functional as TF
import random
from utils import load_image, downsample, psnr, show_images

# ----------------------------
# CONFIG
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = 2

# ----------------------------
# 1. AUGMENTARE + PATCH-URI
# ----------------------------

def extract_augmented_patches(img, patch_height, stride=32):
    if img.dim() == 3:
        img = img.unsqueeze(0)

    patch_width = int(patch_height * 4 / 3)
    patch_width = patch_width - (patch_width % 2)
    b, c, h, w = img.shape

    patches = []
    for i in range(0, h - patch_height + 1, stride):
        for j in range(0, w - patch_width + 1, stride):
            patch = img[0, :, i:i + patch_height, j:j + patch_width]

            if random.random() < 0.2:
                patch = TF.hflip(patch)
            if random.random() < 0.3:
                patch = TF.vflip(patch)
            if random.random() < 0.5:
                patch = TF.rotate(patch, 180)

            patches.append(patch)

    return torch.stack(patches) if patches else torch.empty(0, c, patch_height, patch_width)



def visualize_patches(patches, num_patches=10, title="Visualizing Patches"):
    if patches.size(0) == 0:
        return

    indices = torch.randperm(patches.size(0))[:num_patches]
    selected_patches = patches[indices]

    cols = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float)).item())
    rows = math.ceil(num_patches / cols)

    plt.figure(figsize=(cols * 3, rows * 3))
    plt.suptitle(title, fontsize=16)

    for i, patch in enumerate(selected_patches):
        patch_np = patch.permute(1, 2, 0).cpu().numpy()
        patch_np = torch.clamp(torch.from_numpy(patch_np), 0, 1).numpy()

        plt.subplot(rows, cols, i + 1)
        plt.imshow(patch_np)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ----------------------------
# 2. MODEL
# ----------------------------
class EfficientSRCNN(nn.Module):
    def __init__(self):
        super(EfficientSRCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.clamp(self.net(x), 0.0, 1.0)


# ----------------------------
# 3. PREGĂTIRE
# ----------------------------
model = EfficientSRCNN().to(device)
print("Număr total parametri:", sum(p.numel() for p in model.parameters()))

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon ** 2))

criterion = CharbonnierLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 30-50 1e-3
# 30-50 1e-4
# 30-50 1e-5
train_img = load_image("data/train.jpg", resize_to=(512, 768))
val_img = load_image("data/validation.jpg")

patch_size = 128
patches = extract_augmented_patches(train_img, patch_size, stride=32)
print(f"Patch-uri extrase și augmentate: {patches.shape[0]}")
# Visualize the first 16 patches from the extracted patches
visualize_patches(patches, num_patches=16, title="Augmented Patches")

# ----------------------------
# 4. ANTRENARE
# ----------------------------
batch_size = 32
training_phases = [
    {'lr': 1e-3, 'epochs': 50},
    {'lr': 1e-4, 'epochs': 40},
    {'lr': 1e-5, 'epochs': 30}
]

total_epochs = sum(phase['epochs'] for phase in training_phases)

for phase, config in enumerate(training_phases):
    print(f"\nStarting training phase {phase + 1} with lr={config['lr']} for {config['epochs']} epochs")

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['lr']

    for epoch in range(config['epochs']):
        model.train()
        perm = torch.randperm(patches.size(0))
        total_loss = 0.0

        for i in range(0, patches.size(0), batch_size):
            idx = perm[i:i + batch_size]
            tgt_batch = patches[idx]

            inputs = []
            for patch in tgt_batch:
                patch = patch.unsqueeze(0)
                _, _, h, w = patch.shape
                down = F.interpolate(patch, scale_factor=1 / scale, mode='bicubic', align_corners=False)
                up = F.interpolate(down, size=(h, w), mode='bicubic', align_corners=False)
                inputs.append(up.squeeze(0))

            inp_batch = torch.stack(inputs).to(device)
            output = model(inp_batch)
            loss = criterion(output, tgt_batch.to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss.item()

        # Print loss every 5 epochs
        current_epoch = sum(p['epochs'] for p in training_phases[:phase]) + epoch + 1
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / (patches.size(0) / batch_size)
            print(f"Phase {phase + 1} - Epoch [{current_epoch}/{total_epochs}] Loss: {avg_loss:.6f}")

# ----------------------------
# 5. SALVARE MODEL
# ----------------------------
model_path = f"models/model_x{scale}.pth"
torch.save({
    'epoch': total_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, model_path)

print(f"\nModel salvat în: {model_path}")

# ----------------------------
# 6. VALIDARE
# ----------------------------
scales = [2, 4, 8, 16, 32]

print("\nVALIDARE CU MODEL X2 APLICAT RECURSIV:\n")
model.eval()
for s in scales:
    # 1. Downsample originalul la rezoluție joasă (simulate input)
    val_down, _ = downsample(val_img, s)  # e.g., la 1/8 din dimensiune

    # 2. Upsample treptat cu bicubic + model x2
    inp = val_down.to(device)
    steps = int(torch.log2(torch.tensor(s)).item())  # câte ori aplicăm x2

    with torch.no_grad():
        for _ in range(steps):
            # Bicubic la dimensiunea dublă
            _, _, h, w = inp.shape
            inp = F.interpolate(inp, scale_factor=2, mode='bicubic', align_corners=False)
            # Trecem prin modelul nostru x2
            inp = model(inp)

    # 3. Clamp și comparăm cu originalul la rezoluția full
    val_output = inp.clamp(0, 1)
    val_psnr = psnr(val_output, val_img)
    print(f"Downsampled by {s}x -> PSNR: {val_psnr:.2f} dB")
    show_images(val_img, val_output, f"Reconstruit x{s}")