import os
import sys
import numpy as np
import torch
from collections import defaultdict
from typing import Optional, Sequence, Union
from mmengine.dataset import Compose
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.transforms import LoadImageFromFile
sys.path.insert(0,"/home/ps/CZX/mmsegmentation_czx")
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.apis.inference import init_model
from mmseg.models import BaseSegmentor


ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]


def _preprare_data(imgs: ImageType, model: BaseSegmentor, label:str):

    cfg = model.cfg
    # for t in cfg.test_pipeline:
    #     if t.get('type') == 'LoadAnnotations':
    #         cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        if isinstance(img, np.ndarray):
            data_ = dict(img=img)
        else:
            data_ = dict(img_path=img,
                         seg_map_path=label,
                         reduce_zero_label=True,
                         seg_fields=[])
        
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])


    return data, is_batch



def main():
    config_path = "/home/ps/CZX/mmsegmentation_czx/work_dirs/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention/20231113-120633/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention.py"
    pth_path = "/home/ps/CZX/mmsegmentation_czx/work_dirs/segnext_mscan_-l_2xb4-adamw-focal_loss-40k_seafog_3band-600*600_neck_channel_attention/20231113-120633/iter_40000.pth"
    images_path = "/home/ps/CZX/mmsegmentation_czx/data/seafog_600/crop_image_3band/val"
    labels_path = "/home/ps/CZX/mmsegmentation_czx/data/seafog_600/crop_mask/val"
    images_list = os.listdir(images_path)
    # config = Config.fromfile(config_path)
    # config.work_dir = "/home/ps/CZX/mmsegmentation_czx/work_dirs/grad_CAM_test"
    # data_preprocessor = config.model.data_preprocessor
    # config.model.data_preprocessor = None
    # runner = Runner.from_cfg(config_)
    model = init_model(config=config_path, checkpoint=pth_path)
    # model = runner.build_model(config.model)
    # model =runner.load_checkpoint(pth_path)
    for item in images_list:
        png = item.split('.')[0]+'.png'
        image_path = os.path.join(images_path, item)
        label_path = os.path.join(labels_path, png)
        data, _ = _preprare_data(image_path, model=model, label=label_path)
        # with torch.no_grad():
        # 会自己执行data_preprocessor 不需要再去额外实现
        results = model.test_step(data)
        print('s')
        # data = dict(img_path=image_path)
        # Load = LoadImageFromFile()
        # image = Load.transform(data)
        # data_preprocess = SegDataPreProcessor(mean=data_preprocessor.mean,
        #                                       std = data_preprocessor.std,
        #                                       bgr_to_rgb=data_preprocessor.bgr_to_rgb,
        #                                       pad_val=data_preprocessor.pad_val,
        #                                       seg_pad_val=data_preprocessor.seg_pad_val,
        #                                       size=data_preprocessor.size,
        #                                       test_cfg=data_preprocessor.test_cfg)
        # input = data_preprocess(image)
        print('8')

    
    

if __name__ == "__main__":
    main()