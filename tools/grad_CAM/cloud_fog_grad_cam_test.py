import os
import sys
from mmengine.config import Config
from mmengine.runner import Runner
from mmcv.transforms import LoadImageFromFile
sys.path.insert(0,"/home/ps/CZX/mmsegmentation_czx")
from mmseg.models.data_preprocessor import SegDataPreProcessor
def main():
    config_path = "/home/ps/CZX/mmsegmentation_czx/configs/segnext/segnext_mscan-l_8xb4-adamw-160k_seafog-512*512.py"
    pth_path = "/home/ps/CZX/mmsegmentation_czx/work_dirs/grad_CAM_test/iter_160000.pth"
    images_path = "/home/ps/CZX/mmsegmentation_czx/data/grad_CAM/imgs"
    labels_path = "/home/ps/CZX/mmsegmentation_czx/data/grad_CAM/masks"
    images_list = os.listdir(images_path)
    config = Config.fromfile(config_path)
    config.work_dir = "/home/ps/CZX/mmsegmentation_czx/work_dirs/grad_CAM_test"
    data_preprocessor = config.model.data_preprocessor
    config.model.data_preprocessor = None
    runner = Runner.from_cfg(config)
    # dataset = r
    model = runner.build_model(config.model)
    # model =runner.load_checkpoint(pth_path)
    for item in images_list:
        image_path = os.path.join(images_path, item)
        data = dict(img_path=image_path)
        Load = LoadImageFromFile()
        image = Load.transform(data)
        # data_preprocess = SegDataPreProcessor(mean=data_preprocessor.mean,
        #                                       std = data_preprocessor.std,
        #                                       bgr_to_rgb=data_preprocessor.bgr_to_rgb,
        #                                       pad_val=data_preprocessor.pad_val,
        #                                       seg_pad_val=data_preprocessor.seg_pad_val,
        #                                       size=data_preprocessor.size,
        #                                       test_cfg=data_preprocessor.test_cfg)
        # input = data_preprocess(image)
        outpus = model(image)
        print('8')

    
    

if __name__ == "__main__":
    main()