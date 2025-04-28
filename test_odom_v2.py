import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from depth_to_pcd import depth_to_pcd
from get_normal import get_normal

from hsv import hsv_img

DIR = "/home/daoxin/scratchdata/processed/stair4_filtered"
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

# Open csv
odom = []
with open(os.path.join(DIR, "pose.csv"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        odom.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])])
odom = np.array(odom)

for INDEX in range(10,1000):
    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = Image.open(os.path.join(DIR,f"rgb/{INDEX}.png"))
    image = np.array(image)
    depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
    depth = np.array(depth)/1000

    # Convert depth to point cloud

    points, index = depth_to_pcd(depth, INTRINSICS)

    # Find distnace of pts
    normal = [odom[INDEX, 0], odom[INDEX, 1], odom[INDEX, 2]]
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    print(normal)

    # Find img normal
    img_normal = get_normal(depth, INTRINSICS)
    img_normal_pos = img_normal.reshape(-1, 3)
    img_normal_neg = -img_normal_pos
    dot1 = np.dot(img_normal_pos, normal).reshape(-1, 1)
    dot2 = np.dot(img_normal_neg, normal).reshape(-1, 1)
    
    dot = np.concatenate((dot1, dot2), axis=1)
    index = np.argmax(dot, axis=1)
    img_normal = np.zeros_like(img_normal_pos)
    img_normal[index == 0] = img_normal_pos[index == 0]
    img_normal[index == 1] = img_normal_neg[index == 1]

    if False:
        # Plot normal in 3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        bound = 0.9
        ax.scatter(img_normal[(np.dot(img_normal, normal) < bound) & (points[:,2]!=0), 0], img_normal[(np.dot(img_normal, normal) < bound) & (points[:,2]!=0), 1], img_normal[(np.dot(img_normal, normal) < bound) & (points[:,2]!=0), 2], marker='o', s=1, c='b')
        ax.scatter(img_normal[(np.dot(img_normal, normal) > bound) & (points[:,2]!=0), 0], img_normal[(np.dot(img_normal, normal) > bound) & (points[:,2]!=0), 1], img_normal[(np.dot(img_normal, normal) > bound) & (points[:,2]!=0), 2], marker='o', s=1, c='r')
        plt.show()
        
        img_normal_rgb = (img_normal + 1)/2 * 255
        img_normal_rgb = img_normal_rgb.astype(np.uint8)
        H, W = depth.shape
        img_normal_rgb = img_normal_rgb.reshape(H, W, 3)
        plt.imsave("normal.png", img_normal_rgb)

    # Find Lie algebra of img_normal about normal
    lie = np.zeros((3, 3))
    

    break