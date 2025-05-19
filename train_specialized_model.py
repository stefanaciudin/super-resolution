import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import random
from utils import load_image, downsample, psnr, show_images

# ----------------------------
# CONFIG
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scale = 4

# ----------------------------
# 1. AUGMENTARE + PATCH-URI
# ----------------------------
def extract_augmented_patches(img, patch_size=64, stride=32):
    b, c, h, w = img.shape
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(-1, c, patch_size, patch_size)

    aug_patches = []
    for p in patches:
        if random.random() < 0.5:
            p = TF.hflip(p)
        if random.random() < 0.5:
            p = TF.vflip(p)
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            p = TF.rotate(p, angle)
        aug_patches.append(p)

    return torch.stack(aug_patches)

# ----------------------------
# 2. MODEL
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_img = load_image("data/train.jpg", resize_to=(512, 768))
val_img = load_image("data/validation.jpg")

patches = extract_augmented_patches(train_img, patch_size=64, stride=32)
print(f"Patch-uri extrase și augmentate: {patches.shape[0]}")

# ----------------------------
# 4. ANTRENARE
# ----------------------------
epochs = 2000
batch_size = 32

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(patches.size(0))
    total_loss = 0.0

    for i in range(0, patches.size(0), batch_size):
        idx = perm[i:i+batch_size]
        tgt_batch = patches[idx]

        inputs = []
        for patch in tgt_batch:
            patch = patch.unsqueeze(0)
            down = F.interpolate(patch, scale_factor=1/scale, mode='bicubic', align_corners=False)
            up = F.interpolate(down, size=(64, 64), mode='bicubic', align_corners=False)
            inputs.append(up.squeeze(0))

        inp_batch = torch.stack(inputs).to(device)

        output = model(inp_batch)
        loss = criterion(output, tgt_batch.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / (patches.size(0) / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

# ----------------------------
# 5. SALVARE MODEL
# ----------------------------
model_path = f"models/model_x{scale}.pth"
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, model_path)

print(f"\nModel salvat în: {model_path}")

# ----------------------------
# 6. VALIDARE
# ----------------------------
scales = [2, 4, 8, 16, 32]

print("\nRezultate validare pe diferite scale:")
model.eval()
for s in scales:
    val_down, val_up = downsample(val_img, s)
    with torch.no_grad():
        val_output = model(val_up)
        val_psnr = psnr(val_output, val_img)
        print(f"Downsampled by {s}x -> PSNR: {val_psnr:.2f} dB")
        show_images(val_img, val_output, f"Reconstruit x{s}")