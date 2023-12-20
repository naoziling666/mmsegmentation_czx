import os 
import cv2 as cv
import numpy as np

def calculate_acc_fog(gt_path, pred_path):
    # 这个是用来对整体推测的，推测完将忽略0label的情况考虑了，
    # 即将所有预测结果+1，将land_mask用作0类别
    img_list = os.listdir(pred_path)
    total_num = 0
    correct_num = 0 
    for img in img_list:
        gt = cv.imread(os.path.join(gt_path, img), flags=0)
        pred = cv.imread(os.path.join(pred_path, img), flags=0)
        total_num += np.sum(gt==1)+np.sum(gt==3)
    
        gt_pred = np.logical_and(gt==1, pred==1) | np.logical_and(gt==3, pred==3)
        correct_num += np.sum(gt_pred)
    print(correct_num/total_num)


if __name__ == "__main__":
    gt_path = "/home/ps/CZX/mmsegmentation_czx/data/mask_time_kh_gt_1200"
    pred_path = "/home/ps/CZX/mmsegmentation_czx/work_dirs/mask_time_test/original"

    img_list = os.listdir(pred_path)
    total_num = 0
    correct_num = 0 
    # 在使用tools/test.py 如果打开了reduce_zero_label推理得到的mask即不会包含0，只有其他几个类别
    for img in img_list:
        gt = cv.imread(os.path.join(gt_path, img), flags=0)
        pred = cv.imread(os.path.join(pred_path, img), flags=0)
        total_num += np.sum(gt==1)+np.sum(gt==3)
    
        gt_pred = np.logical_and(gt==1, pred==1) | np.logical_and(gt==3, pred==3)
        correct_num += np.sum(gt_pred)
    print(correct_num/total_num)