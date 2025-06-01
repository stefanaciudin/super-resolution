from pathlib import Path

import torch
import torch.nn as nn
from utils import load_image, downsample, show_images, psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# MODELUL
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


# ----------------------------
# TESTAREA PE IMAGINI DIN FOLDERUL TEST
# ----------------------------

def process_test_images(test_dir="data/test", scale=2, save_results=True):
    test_dir = Path(test_dir)
    results_dir = Path("results")
    if save_results and not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
    valid_extensions = ['.jpg']

    # Get all image files from the test directory
    image_files = [f for f in test_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in valid_extensions]

    if not image_files:
        print(f"No image files found in {test_dir}")
        return

    print(f"\nProcessing {len(image_files)} test images from {test_dir} with scale factor {scale}:")

    for img_path in image_files:
        print(f"Processing {img_path.name}...")
        try:
            test_img = load_image(str(img_path))
            down, up = downsample(test_img, scale)

            with torch.no_grad():
                output = model(up)
                score = psnr(output, test_img)
                print(f"  PSNR: {score:.2f} dB")
                show_images(test_img, output, f"Test: {img_path.name} (x{scale})")

                if save_results:
                    from torchvision.utils import save_image
                    result_grid = torch.cat([test_img, up, output], dim=0)
                    save_path = results_dir / f"sr_{img_path.stem}_x{scale}{img_path.suffix}"
                    save_image(result_grid, str(save_path))
                    print(f"  Results saved to {save_path}")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

for scale in [2, 4, 8, 16, 32]:
    process_test_images(scale=scale, save_results=False)

