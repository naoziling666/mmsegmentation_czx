import argparse
import os
import sys
sys.path.insert(0,"/home/ps/CZX/mmsegmentation_czx")
from mmseg.apis.inference import init_model,inference_model
import cv2 as cv
import numpy as np
import time
import warnings
# def calculate_miou(label, pred, class_num=3):
#     n = class_num
#     pred, label = pred.flatten(), label.flatten()

#     bin_count = np.bincount(n * label + pred, minlength=n ** 2)

#     hist = bin_count[:n ** 2].reshape(n, n)
#     print(hist)
#     iou_per_class = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

#     return np.nanmean(iou_per_class)

def main():
    images_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/snow_ice_data/image/val'

    labels_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/snow_ice_data/label/val'
    save_path = '/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/ice_snow/vis_oridata'
    os.makedirs(save_path, exist_ok=True)

    colors = [[255,255,255],[0,0,255],[255, 0, 0]]
    img_list = os.listdir(labels_path)


    for img in img_list:

        image = cv.imread(os.path.join(images_path, img))
        label = cv.imread(os.path.join(labels_path, img), flags=0)

        # miou_per_sample = calculate_miou(label,pred,3)

        vis_label = np.zeros((512,512,3))
        for i,color in enumerate(colors):
            vis_label[label==i,:]= color
        h, w, c = vis_label.shape
        canvas = np.zeros((h,w*2+20,c))
        canvas[0:h,0:w,:] = image
        canvas[0:h,w+20:w*2+20,:] = vis_label


        cv.imwrite(os.path.join(save_path, img.split('.')[0]+'.png'), canvas)





if __name__ == "__main__":
    main()