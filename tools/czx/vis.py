import os
import cv2 as cv
import numpy as np
baseline_path = "/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/vis/3band_segnext_baseline"
our_model_path = "/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/vis/3band_segnext"
label_path = "/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/origin_labels_kh_1200/val"
save_path = "/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/vis/duibi"
image_list = os.listdir(baseline_path)
for image in image_list:
    baseline_image = cv.imread(os.path.join(baseline_path, image))
    our_model_image = cv.imread(os.path.join(our_model_path, image))
    label_image = cv.imread(os.path.join(label_path, image))
    h,w,c = label_image.shape
    canvas = np.zeros((h,w*3,c))
    canvas[:,0:w,:] = baseline_image
    canvas[:,w:2*w,:] = our_model_image
    canvas[:,2*w:3*w,:] = label_image
    cv.imwrite(os.path.join(save_path, image), canvas)
# image = cv.imread("/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/origin_labels_kh_1200/val/201804210440.png")
# image_g = cv.imread("/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/origin_labels_kh_1200/val/201804210440.png", flags=0)
# for i in range(1200):
#     for j in range(1200):
#         if image_g[i][j]!=0 and image_g[i][j]!=255 and image_g[i][j]!=76:
#             print(image[i][j])
#             print(image_g[i][j])


# print('a')
# a = np.zeros((100,100,3))
# for i in range(100):
#     for j in range(100):
#         a[i][j] = [0,255,255]
# cv.imwrite('/root/autodl-pub/CZX/mmsegmentation_czx/a.png', a)
