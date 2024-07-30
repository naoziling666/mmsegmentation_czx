import os
import numpy as np

img_dir = "/aipt/CZX/mmsegmentation_czx/data/seafog_data/seafog_multiband_600/image/train"
img_list = os.listdir(img_dir)
mean = np.array([0.0 for i in range(9)])
std = np.array([0.0 for i in range(9)])
for img in img_list:
    img_path = os.path.join(img_dir, img)
    img = np.load(img_path)
    mean_single = [img[:,:,i].mean() for i in range(9)]
    std_single = [img[:,:,i].std() for i in range(9)]
    mean += mean_single
    std+=std_single
mean = [i/len(img_list) for i in mean]
std = [i/len(img_list) for i in std]
print(mean)
print(std)