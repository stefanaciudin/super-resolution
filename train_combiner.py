import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from combined_model import CombinedTrainableSRModel
from utils import load_image

# ----------------------------
# SETUP
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# FUNCTII UTILE
# ----------------------------

def extract_patches(img, patch_size=64, stride=32):
    b, c, h, w = img.shape
    patches = img.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.contiguous().view(-1, c, patch_size, patch_size)
    return patches

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon ** 2))

# ----------------------------
# ÎNCĂRCARE DATE
# ----------------------------

train_img = load_image("data/train.jpg", resize_to=(512, 768))
patches = extract_patches(train_img, patch_size=64, stride=32)
print(f"Patch-uri extrase: {patches.shape[0]}")

# ----------------------------
# ÎNCĂRCARE MODELE ȘI COMBINATOR
# ----------------------------

paths = [
    "models/model_x2.pth",
    "models/model_x4.pth",
    "models/model_x8.pth",
    "models/model_x16.pth",
    "models/model_x32.pth"
]

model = CombinedTrainableSRModel(paths).to(device)
criterion = CharbonnierLoss()
optimizer = torch.optim.Adam(model.combiner.parameters(), lr=1e-4)

# ----------------------------
# ANTRENARE STRAT COMBINATOR
# ----------------------------

epochs = 2000
batch_size = 32
scales = [2, 4, 8, 16, 32]

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(patches.size(0))
    total_loss = 0.0

    for i in range(0, patches.size(0), batch_size):
        idx = perm[i:i+batch_size]
        tgt_batch = patches[idx]
        inputs = []

        for patch in tgt_batch:
            scale = np.random.choice(scales)
            patch = patch.unsqueeze(0)
            down = F.interpolate(patch, scale_factor=1/scale, mode='bicubic', align_corners=False)
            up = F.interpolate(down, size=(64, 64), mode='bicubic', align_corners=False)
            inputs.append(up.squeeze(0))

        inp_batch = torch.stack(inputs).to(device)

        output = model(inp_batch)
        loss = criterion(output, tgt_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        avg_loss = total_loss / (patches.size(0) / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

# ----------------------------
# SALVARE MODEL COMBINAT
# ----------------------------

torch.save(model.state_dict(), "models/combined_model_trained.pth")
print("Model combinat antrenat salvat ca 'combined_model_trained.pth'")
