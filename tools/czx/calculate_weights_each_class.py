import cv2 as cv
import os
import numpy as np

root = "/mnt/workspace/users/chengzhixiang/mmsegmentation_czx/data/seafog_multiband/ann_dir/train"
images_list = os.listdir(root)
weights = []
index = 0
for image_path in images_list:
    img = cv.imread(os.path.join(root, image_path), flags=0)

    total_num = 512*512
    num_0 = np.count_nonzero(img == 0)
    num_1 = np.count_nonzero(img == 1)
    num_2 = np.count_nonzero(img == 2)
    num_3 = np.count_nonzero(img == 3)
    num_4 = np.count_nonzero(img == 4)
    if num_4+num_3+num_2+num_1+num_0!=total_num:
        break
    weight_0 = num_0/total_num
    weight_1 = num_1/total_num
    weight_2 = num_2/total_num
    weight_3 = num_3/total_num
    weight_4 = num_4/total_num
    weight_per_image = [weight_0, weight_1, weight_2, weight_3, weight_4]
    weights.append(weight_per_image)
    if(index%50==0):
        print(index)
    index+=1
print(np.mean(weights, axis=0))


