import os
import numpy as np
import cv2 as cv
img_dir = "/root/autodl-fs/mmsegmentation_czx/data/seafog_data/seafog_multiband_600/optical_flow_raft/val"
in_channel = 3
img_list = os.listdir(img_dir)
mean = np.array([0.0 for i in range(in_channel)])
std = np.array([0.0 for i in range(in_channel)])
for img in img_list:
    img_path = os.path.join(img_dir, img)
    if img.endswith(".npy"):
        img = np.load(img_path)
    else:
        img = cv.imread(img_path)
    mean_single = [img[:,:,i].mean() for i in range(in_channel)]
    std_single = [img[:,:,i].std() for i in range(in_channel)]
    mean += mean_single
    std+=std_single
mean = [i/len(img_list) for i in mean]
std = [i/len(img_list) for i in std]
print(mean)
print(std)