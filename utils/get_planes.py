import numpy as np
from utils.get_normal import get_normal
from matplotlib import pyplot as plt

def get_planes(depth, INTRINSICS, ANGLE_CLUSTER):
    # Output mask
    mask = np.zeros_like(depth, dtype=np.uint8)

    W, H = depth.shape
    ANGLE_INCREMENT = 41
    KERNEL_2D = 5

    normal = get_normal(depth, INTRINSICS)
    
    img_normal_angle = np.zeros((W, H, 2))
    img_normal_angle[:, :, 0] = np.arctan(normal[:,:,0]/(normal[:,:,2]+1e-15))
    img_normal_angle[:, :, 1] = np.arctan(normal[:,:,1]/(normal[:,:,2]+1e-15))

    angle_cluster = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    for i in range(0, ANGLE_INCREMENT):
        for j in range(0, ANGLE_INCREMENT):
            angle_cluster[i, j] = np.sum((img_normal_angle[:, :, 0] >= (i-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT ) 
                                         & (img_normal_angle[:, :, 0] < (i+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] >= (j-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (img_normal_angle[:, :, 1] < (j+1-ANGLE_INCREMENT/2)*np.pi/ANGLE_INCREMENT)
                                         & (depth != 0))

    dillation = np.zeros((ANGLE_INCREMENT, ANGLE_INCREMENT))
    angle_cluster = np.pad(angle_cluster, ((KERNEL_2D//2, KERNEL_2D//2), (KERNEL_2D//2, KERNEL_2D//2)), mode='wrap')
    for i in range(ANGLE_INCREMENT):
        for j in range(ANGLE_INCREMENT):
            dillation[i, j] = np.max(angle_cluster[i:i+KERNEL_2D, j:j+KERNEL_2D])

    angle_cluster = angle_cluster[KERNEL_2D//2:-KERNEL_2D//2+1, KERNEL_2D//2:-KERNEL_2D//2+1]
    
    # Find index where angle_cluster is equal to dillation
    index = np.where(angle_cluster == dillation)

    # Get the best peaks in the angle_cluster
    best_peaks = np.argsort(angle_cluster[index])[-ANGLE_CLUSTER:]
    print(best_peaks)
    best_peaks_index = np.array([index[0][best_peaks], index[1][best_peaks]]).T

    for i in range(best_peaks_index.shape[0]):
        print(angle_cluster[best_peaks_index[i][0], best_peaks_index[i][1]])
        angle_x = (best_peaks_index[i][0] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        angle_y = (best_peaks_index[i][1] - ANGLE_INCREMENT//2) * np.pi / ANGLE_INCREMENT
        
        grav_normal = np.array([np.tan(angle_x), np.tan(angle_y), 1])
        grav_normal = grav_normal / np.linalg.norm(grav_normal)

        img_normal_pos = img_normal.reshape(-1, 3)
        img_normal_neg = -img_normal_pos
        dot1 = np.dot(img_normal_pos, grav_normal).reshape(-1, 1)
        dot2 = np.dot(img_normal_neg, grav_normal).reshape(-1, 1)
        
        dot = np.concatenate((dot1, dot2), axis=1)
        normal_index = np.argmax(dot, axis=1)
        img_normal = np.zeros_like(img_normal_pos)
        img_normal[normal_index == 0] = img_normal_pos[normal_index == 0]
        img_normal[normal_index == 1] = img_normal_neg[normal_index == 1]

        dot_bound = 0.9
        correction_iteration = 10
        grav_normal = gravity_correction(grav_normal,img_normal.reshape(-1,3), points.reshape(-1,3), dot_bound, correction_iteration)

        if True:
            dot1 = np.dot(img_normal, grav_normal).reshape(-1,1)
            dot2 = np.dot(img_normal, -grav_normal).reshape(-1,1)

            angle_dist = np.concatenate((dot1, dot2), axis=1)
            angle_dist = np.max(angle_dist, axis=1)
            scalar_dist = np.dot(points.reshape(-1,3), grav_normal)
            scalar_dist[angle_dist < dot_bound] = 0
            scalar_dist[points.reshape(-1,3)[:, 2] == 0] = 0

            # Plot histogram
            fig, ax = plt.subplots()
            ax.hist(scalar_dist[scalar_dist!=0], bins=1000)
            plt.xlabel("Distance")
            plt.ylabel("Count")
            plt.title("Histogram of Distance")
            fig.savefig("./histogram.png")

        kernel_size = 11
        cluster_size = 11

        mask_2d = get_mask(grav_normal, img_normal.reshape(-1,3), points.reshape(-1,3), dot_bound, kernel_size, cluster_size, plane_cnt=1)
        mask_2d = mask_2d.reshape(W, H)
        cur_plane = mask.max()
        mask = np.where(mask_2d != 0, mask_2d + cur_plane, mask)
