_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/seafog_multiband.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 115.23, 117.22, 118.22, 113.29, 110.23, 112.92],
    std=[58.395, 57.12, 57.375, 57.275, 57.485, 57.565, 57.585, 57.590, 58.005],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))
model = dict(
    in_channels=9,
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=5))
