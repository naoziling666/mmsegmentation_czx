import argparse
import os
import sys
sys.path.insert(0,"/home/ps/CZX/mmsegmentation_czx")
from mmseg.apis.inference import init_model,inference_model
import cv2 as cv
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")
def parse_args():
    parser = argparse.ArgumentParser(description='inferce whole image')
    parser.add_argument('--config', help='the config path of model',default="/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600/20240303-224202/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600.py")
    parser.add_argument('--checkpoint', help='the checkpoint path of model', default="/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600/20240303-224202/iter_40000.pth")
    parser.add_argument('--images_path', help='the path of the image that requires inference', default="/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/origin_images_kh_1200/val")
    parser.add_argument('--vis_label_path', help='the path of the vis_label(gt)', default="/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/origin_labels_kh_1200/val")
    parser.add_argument('--vis_save_path', help='the path to save result(vis) of inference', default="/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/vis/channel_attention")
    parser.add_argument('--mask_save_path', help='the path to save result(mask) of inference', default="/root/autodl-pub/CZX/mmsegmentation_czx/work_dirs/mask_pred/3band_segnext_baseline")
    parser.add_argument('--land_gt_path', help='the path to use the mask of land', default="/root/autodl-pub/CZX/mmsegmentation_czx/data/seafog_data/land_mask_kh_1200/val")
    
    args = parser.parse_args()


    return args

def inference_img_patch(model, img_patch):
    pred = inference_model(model, img_patch)
    pred = np.array(pred.pred_sem_seg.data.cpu().squeeze())
    return pred
def main():
    
    args = parse_args()
    if not os.path.exists(args.vis_save_path):
        os.mkdir(args.vis_save_path)
    if not os.path.exists(args.mask_save_path):
        os.mkdir(args.mask_save_path)
    colors = [[0, 0, 0], [0, 0, 255], [0,255,0], [0,255,255],[255, 255, 255]]
    img_list = os.listdir(args.images_path)
    model = init_model(config=args.config, checkpoint=args.checkpoint)
    print(len(img_list))
    start = time.time()
    for img in img_list:
        image = np.load(os.path.join(args.images_path, img))
        land_gt = cv.imread(os.path.join(args.land_gt_path, img.split('.')[0]+'.png'), flags=0)
        vis_label = cv.imread(os.path.join(args.vis_label_path, img.split('.')[0]+'.png'))
        pred_0_0 = inference_img_patch(model, image[0:600, 0:600, :3])
        pred_0_1 = inference_img_patch(model, image[0:600, 600:1200, :3])
        pred_1_0 = inference_img_patch(model, image[600:1200, 0:600, :3])
        pred_1_1 = inference_img_patch(model, image[600:1200, 600:1200, :3])
        pred = np.zeros((1200,1200))
        pred[0:600, 0:600] = pred_0_0
        pred[0:600, 600:1200] = pred_0_1
        pred[600:1200, 0:600] = pred_1_0
        pred[600:1200, 600:1200] = pred_1_1
        # 因为在训练时忽略了0的label
        pred = pred+1
        # use the mask of land
        pred[land_gt==0] = 0

        vis = np.zeros((1200,1200,3))
        for i,color in enumerate(colors):
            vis[pred==i,:]= color
        h, w, c = vis_label.shape
        canvas = np.zeros((h,w*2,c))
        canvas[0:h,0:w,:] = vis_label
        canvas[0:h,w:w*2,:] = vis
        # cv.imwrite(os.path.join(args.mask_save_path, img.split('.')[0]+'.png'), pred)
        cv.imwrite(os.path.join(args.vis_save_path, img.split('.')[0]+'.png'), vis)
    end = time.time()
    print("inference single image spend {} seconds".format(end-start))
        


if __name__ == "__main__":
    main()