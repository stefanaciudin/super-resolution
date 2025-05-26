import torch
from combined_model import CombinedTrainableSRModel
from utils import downsample, load_image, psnr, show_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = [
    "models/model_x2.pth",
    "models/model_x4.pth",
    "models/model_x8.pth",
    "models/model_x16.pth",
    "models/model_x32.pth"
]

model = CombinedTrainableSRModel(paths).to(device)
model.load_state_dict(torch.load("models/combined_model_trained.pth", map_location=device))
model.eval()

# --------------------------------------------------
# TESTARE PE IMAGINEA DE VALIDARE
# --------------------------------------------------

img_path = "data/validation.jpg"
img = load_image(img_path)

scales = [2, 4, 8, 16, 32]

for scale in scales:
    _, input_img = downsample(img, scale)

    with torch.no_grad():
        output = model(input_img)

    score = psnr(output, img)
    print(f"{img_path}, scale: {scale}-> PSNR: {score:.2f} dB")

    output_path = f"output_combined_x{scale}.png"
    show_images(img, output, f"Super-Resolutie x{scale}")