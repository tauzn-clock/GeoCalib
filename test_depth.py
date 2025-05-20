import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import torch
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

from utils.depth_to_pcd import depth_to_pcd
from utils.get_normal import get_normal
from utils.gravity_correction import gravity_correction
from utils.get_mask import get_mask
from utils.hsv import hsv_img
from utils.metric3d import metric3d
from utils.test_depth import parabolic, flat

from Metric3D.run_metric3d import run_metric3d

#from scipy.spatial.transform import Rotation as R

DIR = "/scratchdata/stair3"
with open(os.path.join(DIR, "camera_info.json"), "r") as f:
    camera_info = json.load(f)
INTRINSICS = camera_info["P"]
print(INTRINSICS)

USE_MEASURED = True
USE_ORIENTATION = True
ANGLE_INCREMENT = 41
KERNEL_2D = 5

#model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
#model = model.cuda() if torch.cuda.is_available() else model

## Open csv
#odom = []
#with open(os.path.join(DIR, "pose.csv"), "r") as f:
#    lines = f.readlines()
#    for line in lines:
#        line = line.strip().split(",")
#        odom.append([float(x) for x in line])
#odom = np.array(odom)

for INDEX in range(0,1000):
    # load image as tensor in range [0, 1] with shape [C, H, W]
    image = Image.open(os.path.join(DIR,f"rgb/{INDEX}.png"))
    image = np.array(image)
    if USE_MEASURED:
        depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
        depth = np.array(depth)/1000
        W, H = depth.shape
        #depth = flat(H,W)
        img_normal = get_normal(depth, INTRINSICS)
    else:
        depth = Image.open(os.path.join(DIR, f"depth/{INDEX}.png"))
        depth = np.array(depth)/1000
        #_, img_normal = metric3d(model, image)

    output_depth, output_normal = run_metric3d(image)

    plt.imsave('depth.png', output_depth)
    plt.imsave('normal.png', (output_normal+1)/2)

    exit()