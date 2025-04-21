from geocalib import GeoCalib
import torch
import numpy as np
import json
import os
from PIL import Image
from depth_to_pcd import depth_to_pcd

DIR = "/home/daoxin/scratchdata/processed/stair2/"
INDEX = 0
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

# load image as tensor in range [0, 1] with shape [C, H, W]
image = model.load_image(os.path.join(DIR,f"rgb/{INDEX}.png")).to(device)
depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
depth = np.array(depth)/1000

result = model.calibrate(image)

# Convert depth to point cloud

points, index = depth_to_pcd(depth, INTRINSICS)
print(points.shape)
print(index.shape)

print(depth.max(), points[:, 2].max())

# Find distnace of pts

normal = [result["gravity"].x.item(), result["gravity"].y.item(), result["gravity"].z.item()]
normal = np.array(normal)
normal = normal / np.linalg.norm(normal)
print(normal)

dist = np.dot(points, normal)[points[:, 2] > 0]

import matplotlib.pyplot as plt

# Plot dist as histogram
fig, ax = plt.subplots()
plt.hist(dist, bins=100)
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.title("Distance Histogram")
# Save histogram
plt.savefig("dist.png")