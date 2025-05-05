import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

rgb = Image.open('/home/daoxin/scratchdata/processed/stairs_up/rgb/0.png').convert('RGB')
rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
rgb = rgb.cuda() if torch.cuda.is_available() else rgb

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
model = model.cuda() if torch.cuda.is_available() else model
pred_depth, confidence, output_dict = model.inference({'input': rgb})
pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details

# Visualize depth
plt.imsave('depth.png', pred_depth[0, 0].cpu().numpy())
plt.imsave('normal.png', pred_normal[0, 0].cpu().numpy())
plt.imsave('confidence.png', confidence[0, 0].cpu().numpy())
plt.imsave('normal_confidence.png', normal_confidence[0].cpu().numpy())