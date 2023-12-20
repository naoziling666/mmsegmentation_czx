# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class YhDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('fog', 'other'),# 正确的

        # 在label中
        # 0对应紫色，类别为land
        # 1对应青色，类别为fog
        # 2对应黄色，类别为cloud


        palette=[[0, 255, 255], [255,255,0]],
        # 类别索引为 255 的像素，在计算损失时会被忽略

        )
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
