import os
import shutil
import cv2 as cv
import numpy as np
if __name__ == "__main__":
    mask_path = '/home/ps/CZX/mmsegmentation_czx/data/mask_time_kh_pred_select_1200'
    mask_list = sorted(os.listdir(mask_path))

    for i in range(0,len(mask_list),3):
        shutil.copy2(os.path.join(mask_path, mask_list[i+1]),
                     os.path.join('work_dirs/mask_time_test/original', mask_list[i+1]))
        img_1 = cv.imread(os.path.join(mask_path,mask_list[i]), flags=0)
        img_2 = cv.imread(os.path.join(mask_path,mask_list[i+1]), flags=0)
        img_3 = cv.imread(os.path.join(mask_path,mask_list[i+2]), flags=0)
        mask = img_2.copy()
        for j in range(len(img_1)):
            for k in range(len(img_1[0])):
                class_1, class_2, class_3 = img_1[j][k],img_2[j][k],img_3[j][k]
                
                if img_2[j][k]==2 and img_1[j][k]==1:
                    mask[j][k] = 1
                elif img_2[j][k]==2 and img_3[j][k]==1:
                    mask[j][k] = 1
                    
        cv.imwrite(os.path.join('/home/ps/CZX/mmsegmentation_czx/work_dirs/mask_time_test/change', mask_list[i+1]), mask)
                
        