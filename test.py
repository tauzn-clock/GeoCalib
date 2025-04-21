from geocalib import GeoCalib
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

# load image as tensor in range [0, 1] with shape [C, H, W]
image = model.load_image("/home/daoxin/scratchdata/processed/alcove2/rgb/2.png").to(device)
result = model.calibrate(image)

print("camera:", result["camera"])
print("gravity:", result["gravity"])