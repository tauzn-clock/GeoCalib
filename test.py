from geocalib import GeoCalib
import torch
import numpy as np
import json
import os
from PIL import Image
from depth_to_pcd import depth_to_pcd

DIR = "/scratchdata/processed/stair_up"
for INDEX in range(1,200):
    with open(os.path.join(DIR, "camera_info.json"), "r") as f:
        camera_info = json.load(f)
    INTRINSICS = camera_info["P"]
    print(INTRINSICS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib().to(device)

    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = model.load_image(os.path.join(DIR,f"rgb/{INDEX}.png")).to(device)
    depth = Image.open(os.path.join(DIR, f"depth/{INDEX-1}.png"))
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

    dist = np.dot(points, normal)

    if True:
        import matplotlib.pyplot as plt

        # Plot dist as histogram
        fig, ax = plt.subplots()
        plt.hist(dist, bins=100)
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title("Distance Histogram")
        # Save histogram
        plt.savefig("dist.png")

    # Bin dist into bins of size 0.1
    bins = np.arange(dist.min(), dist.max(), 0.1)
    hist, bin_edges = np.histogram(dist, bins=bins)

    # Create mask
    mask = np.zeros_like(depth)
    H, W = depth.shape
    cnt = 1
    for i in range(len(hist)):
        corresponding_index = index[(dist > bin_edges[i]) & (dist <= bin_edges[i+1]) & (dist != 0)]
        if len(corresponding_index) < 0.02 * H * W:
            continue
        mask[corresponding_index[:, 0], corresponding_index[:, 1]] = cnt
        cnt += 1

    print(mask.shape)

    if True:
        import matplotlib.pyplot as plt

        # Plot mask
        fig, ax = plt.subplots()
        ax.imshow(mask)
        plt.axis('off')
        # Save mask
        plt.savefig("mask.png")

        fig, ax = plt.subplots()
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.imshow(mask, alpha=0.5, cmap="hsv")
        plt.axis('off')
        # Save depth
        plt.savefig(os.path.join(DIR, f"test/{INDEX}.png"), bbox_inches='tight', pad_inches=0)