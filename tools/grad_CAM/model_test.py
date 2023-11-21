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


def main():
    config_path = "/home/ps/CZX/mmsegmentation_czx/configs/segnext/segnext_mscan_-l_2xb6-adamw-focal_loss-40k_seafog_3band-512*512._dual_decode.py"
    model = init_model(config=config_path)
    print('w')
    
    
if __name__ == "__main__":
    main()