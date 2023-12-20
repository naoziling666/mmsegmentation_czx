import os
import cv2 as cv
import numpy as np
from mmseg.apis.inference import init_model,inference_model
# npy_path = "/home/ps/CZX/mmsegmentation_czx/data/seafog_600/crop_image_16band/val"
# mask_list = os.listdir(npy_path)
# for item in mask_list:
#     mask = np.load(os.path.join(npy_path, item))
#     print('s')




img_list = os.listdir("/home/ps/CZX/mmsegmentation_czx/data/fy4/image_3band/train")
model = init_model(config="/home/ps/CZX/mmsegmentation_czx/work_dirs/FY4_segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention_cascade_decode/20231211-214830/FY4_segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention_cascade_decode.py",
                   checkpoint="/home/ps/CZX/mmsegmentation_czx/work_dirs/FY4_segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention_cascade_decode/20231211-214830/iter_40000.pth")

for img in img_list:
    pred = inference_model(model, os.path.join("/home/ps/CZX/mmsegmentation_czx/data/fy4/image_3band/train", img))
    print('s')