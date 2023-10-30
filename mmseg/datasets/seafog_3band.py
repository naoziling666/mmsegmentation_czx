# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SeafogDataset_3band(BaseSegDataset):
    """SeafogDataset.

    In segmentation map annotation for Seafog, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('land', 'fog', 'cloud', 'cloud_fog', 'ocean'),# 正确的
        palette=[[0, 0, 0], [233, 0, 0], [0,230,0], [247,186,11],[227, 227, 2227]],
        # 类别索引为 255 的像素，在计算损失时会被忽略
        # label_map = {0:0, 1:1, 2:2, 3:3, 4:4}
        )
    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)