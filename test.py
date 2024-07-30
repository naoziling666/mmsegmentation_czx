import numpy as np
import os
import matplotlib.pyplot as plt
img_path = "/aipt/CZX/mmsegmentation_czx/data/seafog_data/seafog_multiband_600/image/train"
label_path = "/aipt/CZX/mmsegmentation_czx/data/seafog_data/seafog_multiband_600/label/train"
train_images = os.listdir(img_path)
label_images = os.listdir(label_path)
print(len(train_images), len(label_images))

# img_path = "./data/seafog_data/seafog_multiband_600/image/train/202006060520_2_2.npy"
# a = np.load(img_path)
# print(a.shape)