# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Fy4Dataset(BaseSegDataset):
    """SeafogDataset.

    In segmentation map annotation for Seafog, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('fog', 'cloud', 'cloud_fog', 'ocean'),# 正确的
        # 之前写错的classes=('land', 'cloud', 'fog', 'cloud_fog', 'ocean'),
        # 在label中
        # 0对应黑色，类别为land
        # 1对应红色，类别为fog
        # 2对应绿色，类别为cloud
        # 3对应黄色，类别为cloud_fog
        # 4对应白色，类别为ocean

        palette=[[233, 0, 0], [0,230,0], [247,186,11],[227, 227, 2227]],
        # 类别索引为 255 的像素，在计算损失时会被忽略
        # label_map = {0:0, 1:1, 2:2, 3:3, 4:4}
        )
    def __init__(self,
                 img_suffix='.npy',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
