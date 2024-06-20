import numpy as np
import os
import cv2 as cv

if __name__ == "__main__":
    root_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_mask/train'
    img_3band_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_3band/train'
    img_6band_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_6band/train'
    img_16band_path = '/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/seafog_600/crop_image_16band/train'    
    img_list = os.listdir(root_path)
    account = []
    account_all = []
    name = []
    for img_name in img_list:
        img = cv.imread(os.path.join(root_path, img_name), flags=0)
        num_not_background = 360000-np.sum(img==0)
        account_fog_cloud = np.sum(img==3)/num_not_background
        account_fog_cloud_all = np.sum(img==3)/360000
        account.append(account_fog_cloud)
        if account_fog_cloud>=0.2:
            name.append(img_name)
        account_all.append(account_fog_cloud_all)
    account.sort(reverse=True)
    account_all.sort(reverse=True)
    for item in name:
        item = item.split('.')[0]+'.npy'
        os.rename(os.path.join(img_3band_path, item), os.path.join(img_3band_path, 's_'+item))
        os.rename(os.path.join(img_6band_path, item), os.path.join(img_6band_path, 's_'+item))
        os.rename(os.path.join(img_16band_path, item), os.path.join(img_16band_path, 's_'+item))
    for item in name:
        os.rename(os.path.join(root_path, item), os.path.join(root_path, 's_'+item))

    print('sss')