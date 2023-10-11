_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/seafog_multiband.py',
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
    size=crop_size)
# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=9,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512],
                    num_classes=5))
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader


