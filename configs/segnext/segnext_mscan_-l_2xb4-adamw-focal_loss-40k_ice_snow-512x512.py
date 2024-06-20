_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py',
    '../_base_/datasets/ice_snow.py'
]
# model settings
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=(512,512),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.3,
        depths=[3, 5, 27, 3],
        attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
        attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='BN', requires_grad=True)),
    # neck=dict(
    #     type='SceneRelation',
    #     in_channels=512,
    #     channel_list=[64, 128, 320, 512]),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[128, 320, 512],
        in_index=[1, 2, 3],
        channels=1024,
        ham_channels=1024,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)),
    auxiliary_head=dict(
        type='AssymetricDecoder',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        num_classes=3,
        input_transform = 'multiple_select',
        loss_decode=dict(
            type='FocalLoss', use_sigmoid=True, loss_weight=0.6)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
train_dataloader = dict(
    batch_size=6,
    num_workers=6,)
# optimizer

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]



# model_wrapper_cfg = dict(
#                 find_unused_parameters=True
#             )