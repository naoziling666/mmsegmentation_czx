_base_ = [
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]

workflow = [('train', 1)]  ###

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-100, -50, -5.0, 200, 50, 3.0]
point_cloud_range_pinhole = [-100, -50, -5.0, 200, 50, 3.0]
point_cloud_range_fisheye = [-50, -50, -5.0, 50, 50, 3.0]
point_cloud_range_mask2d3d = [
    [-50.0, -20.0, -5.0, 120.0, 20.0, 3.0],  # CONE
    [-50.0, -20.0, -5.0, 70.0, 20.0, 3.0],  # POLE
    [-50.0, -20.0, -5.0, 120.0, 20.0, 3.0],  # ISOLATION_BARREL
]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True)

fov120_2_center_prime = [[0, 0, 1, 2], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]
gop_names = ['CONE', 'POLE', 'ISOLATION_BARREL']
pvb_names = []
class_names = pvb_names+gop_names

cams_pinhole_crop_map = {
    'center_camera_fov20': 'center_camera_fov30',
    'center_camera_fov60': 'center_camera_fov30',
    'center_camera_fov105': 'center_camera_fov120',
}

cams_pinhole_crop_map_test = {
    'center_camera_fov105': 'center_camera_fov120',
}
cams_pinhole_crop = [camera_name for camera_name in cams_pinhole_crop_map]

# cams_pinhole = ['center_camera_fov30','center_camera_fov120', 'left_front_camera', 'left_rear_camera', 'rear_camera', 'right_rear_camera', "right_front_camera", 'center_camera_fov30_1']
## T68
cams_pinhole = ['center_camera_fov30','center_camera_fov120', 'left_front_camera', 'left_rear_camera', 'rear_camera', 'right_rear_camera', "right_front_camera", 'center_camera_fov20']

cams_fisheye = []

cams = cams_pinhole + cams_pinhole_crop

cams_transformer = cams_pinhole

# Model setting
model_setting = '8P'
use_pinhole = True
use_fisheye = False
use_virtual_cam = False

num_cams = len(cams)  ###
num_cams_transformer = len(cams_transformer)  ###

batch_size = 1
num_epochs = 40
num_gpus = 32 # not need
num_iters_per_epoch = 391839 // (num_gpus * batch_size)  # not need

queue_length = 1
num_frame_losses = 1
collect_keys=['ori_intrinsics_pinhole', 'intrinsics_pinhole', 'extrinsics_pinhole', 'lidar2img_pinhole',  
              'ori_intrinsics_pinhole_crop', 'intrinsics_pinhole_crop', 'extrinsics_pinhole_crop', 'lidar2img_pinhole_crop',
              'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='SPetr3DPFMonoNV2d',
    stride=32,
    model_setting=model_setting,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    use_grid_mask=True,
    use_pinhole=use_pinhole,
    use_fisheye=use_fisheye,
    mono_position_level=[0],
    use_nv2d_query_ref_points=True,
    use_nms2d=True,
    nms2d_threshold=[0.5,0.5,0.5],
    merge_mono=False,  # TODO
    use_distance_score=True,  # TODO
    distance_cfg = dict(
        distance = [[0],[0],[0]],
        distace_score = [[0.35,0.3],[0.4,0.35],[0.3,0.25]]),
    img_backbone_pinhole=dict(
        type='DLANetLight',
        depth="34v3",
        in_channels=3,
        out_indices=(5,),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='sh1424hdd:s3://sh1424_hdd_datasets/users/RoadUser/pretrained_models/dla34.pth'
        # )
        ),
    # img_backbone_pinhole_crop=dict(
    #     type='DLANetLight',
    #     depth="34v3",
    #     in_channels=3,
    #     out_indices=(5,),
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     # init_cfg=dict(
    #     #     type='Pretrained',
    #     #     checkpoint='sh1424hdd:s3://sh1424_hdd_datasets/users/RoadUser/pretrained_models/dla34.pth'
    #     # )
    #     ),
    # img_rpn_head=dict(
    #     type='Fcos3dHead',
    #     img_feats_name='img_feats_pinhole_crop',
    #     regress_ranges=[(-1, 256)],
    #     num_classes=len(class_names),
    #     in_channels=128,
    #     stacked_convs=2,
    #     feat_channels=256,
    #     use_direction_classifier=True,
    #     diff_rad_by_sin=True,
    #     pred_attrs=False,
    #     pred_velo=True,
    #     pred_bbox2d=True,
    #     dir_offset=0.7854,  # pi/4
    #     strides=[32],
    #     bbox_code_size=7,
    #     group_reg_dims=(2, 1, 3, 1),  # offset, depth, size, rot, velocity
    #     cls_branch=(256, ),
    #     reg_branch=(
    #         (256, ),  # offset
    #         (256, ),  # depth
    #         (256, ),  # size
    #         (256, ),  # rot
    #     ),
    #     dir_branch=(256, ),
    #     attr_branch=(256, ),
    #     bbox2d_branch=(256, ),
    #     loss_cls=dict(
    #         type='FocalLoss',
    #         use_sigmoid=True,
    #         gamma=2.0,
    #         alpha=0.25,
    #         loss_weight=1.0),
    #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    #     loss_dir=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #     loss_attr=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    #     loss_bbox2d=dict(type='IoULoss', loss_weight=1.0),
    #     loss_centerness=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    #     bbox_coder=dict(
    #         type='FCOS3DBBoxCoder', code_size=7, rescale_depth=True),
    #     norm_on_bbox=True,
    #     centerness_on_reg=True,
    #     center_sampling=True,
    #     conv_bias=True,
    #     dcn_on_last_conv=False,
    #     norm_cfg=None,  # no head bn
    #     rescale_depth=True,
    #     is_deploy=False,
    #     train_cfg=dict(
    #         allowed_border=0,
    #         code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
    #         pos_weight=-1,
    #         debug=False),
    #     rp_cfg=dict(
    #         nms_kernel_size=3,
    #         num_proposals=[32, 32, 32, 32, 32, 32, 32, 32],
    #         pc_range=point_cloud_range,
    #         use_nms_pool=True)
    # ),
    # ----------------------------------------------------------------------------------------------------

    img_rpn_head_pinhole=dict(
        type='PinholeFisheyeQueryRPN',
        regress_ranges=[(-1, 256)],
        num_classes=len(class_names),
        in_channels=128,
        stacked_convs=2,
        feat_channels=256,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        pred_attrs=False,
        pred_velo=False,
        keep_3dbox_bev_range=point_cloud_range_mask2d3d,
        pred_bbox2d=False,
        use_bbox2d_reg_feat=False,
        dir_offset=0.7854,  # pi/4
        strides=[32],
        bbox_code_size=7,
        group_reg_dims=(2, 1, 3, 1),  # offset, depth, size, rot
        cls_branch=(256, ),
        reg_branch=(
            (256, ),  # offset
            (256, ),  # depth
            (256, ),  # size
            (256, )  # rot
        ),
        dir_branch=(256, ),
        attr_branch=(256, ),
        bbox2d_branch=(256, ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox2d=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        bbox_coder=dict(
            type='FCOS3DBBoxCoder', code_size=7, rescale_depth=True),
        norm_on_bbox=True,
        centerness_on_reg=True,
        center_sampling=True,
        conv_bias=True,
        dcn_on_last_conv=False,
        norm_cfg=None,  # no head bn
        rescale_depth=True,
        is_deploy=False,
        train_cfg=dict(
            allowed_border=0,
            code_weight=[1.0, 1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
            pos_weight=-1,
            debug=False),
        rp_cfg=dict(
            nms_kernel_size=3,
            num_proposals=[8, 8, 8, 8, 8, 8, 8, 8],
            pc_range=point_cloud_range,
            use_nms_pool=True)
    ),

    # ----------------------------------------------------------------------------------------------------
    pts_bbox_head=dict(
        type='StreamPETRHeadPF',
        fisheye_distortion_mode='KB',
        cams=cams_transformer,
        num_classes=len(class_names),
        in_channels=128,
        embed_dims=128,
        num_query=64,
        memory_len=128,
        use_single_frame=True,  # TODO, use_single_frame
        topk_proposals=64,
        num_propagated=64,
        depth_num=64,
        with_dn=True,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=8,       # noise groups
        noise_scale = 1.0, 
        dn_weight= 1.0,  # dn loss weight
        split = 0.75,    # positive rate
        LID=True,
        with_position=True,
        position_range=point_cloud_range,
        position_range_pinhole = point_cloud_range_pinhole,
        position_range_fisheye = point_cloud_range_fisheye,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.1],
        use_spatial_alignment=True,
        cone_with_params=False,
        cone_with_coord=True,
        use_pinhole=use_pinhole,
        use_fisheye=use_fisheye,
        use_mask=True,
        use_origin_intrinsic=False,
        init_ref_points=True,
        params_refine_ref_points=False,
        use_scale_nms=dict(
            ENABLE=True,
            RATIO=5.0,),
        transformer=dict(
            type='SPETRTransformerMultiInputs',
            num_layers = 3,
            num_heads = 8,
            embed_dims = 128,
            feedforward_channels= 2048,
            ffn_drop = 0.1,
            dropout = 0.1,
            num_views = num_cams_transformer,
            use_pinhole=use_pinhole,
            use_fisheye=use_fisheye,
            ),
        bbox_coder=dict(
            type='NMSFreeCoder2',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=len(class_names)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='RotatedIoU3DLoss', loss_weight=0.1),
        # loss_reprojection=dict(type='ReprojectionLoss', loss_reprojection_2d=None,loss_reprojection_3d=dict(type='GIoULoss', eps=1e-5, reduction='mean', loss_weight=0.1)),
        loss_reprojection=None,
        ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            bbox_order='xyzwlh',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range),)))

import os
conf_path='{}/petreloss.conf'.format(os.getcwd())
file_client_args_ceph = dict(
    backend='petrel',
    conf_path=conf_path,
    path_mapping=dict({
        "s3://sh1984_datasets/": "sh1984:s3://sh1984_datasets/",
        "s3://sh1424_datasets/": "sh1424:s3://sh1424_datasets/",
        "s3://sh1424_hdd_datasets/": "sh1424hdd:s3://sh1424_hdd_datasets/",
        "s3://shlgssd_autolabel/": "shlgssd:s3://shlgssd_autolabel/",
        "s3://sh1424hdd_autolabel/": "sh1424hdd:s3://sh1424hdd_autolabel/",
        "s3://shlgssd_datasets/": "shlgssd:s3://shlgssd_datasets/",
        "s3://sz20hdd_datasets/": "clasz20:s3://sz20hdd_datasets/",
        "s3://sh41hdd_autolabel_rawdata/": "sh41hdd:s3://sh41hdd_autolabel_rawdata/",
        "s3://1424ssd_autolabel/": "sh1424:s3://1424ssd_autolabel/",
        "s3://sh1424_hdd_datasets/": "sh1424hdd:s3://sh1424_hdd_datasets/",
        "s3://sh1424hdd_autolabel/": "sh1424hdd:s3://sh1424hdd_autolabel/",
        "s3://sh41hdd_autolabel/": "sh41hdd:s3://sh41hdd_autolabel/",
        "s3://sz20hdd_unpack/": "clasz20:s3://sz20hdd_unpack/",
        "s3://bj30hdd_unpack/": "bj30hdd:s3://bj30hdd_unpack/",
        "s3://sdc_adas/": "adsdc:s3://sdc_adas/",
        "s3://sdcoss2_lsm/": "sdcoss2_lsm:s3://sdcoss2_lsm/",
        "s3://sdchdd_datasets/": "clasdchdd:s3://sdchdd_datasets/",
        "s3://sdc_autolabel/": "adsdc:s3://sdc_autolabel/",
        "s3://AVP_BEV/2023_07_vehicle6_nudge_train/": "bj16:s3://AVP_BEV/2023_07_vehicle6_nudge_train/",
        "s3://AVP_BEV/mnt/lustre/share_data/zhongligeng/data/xzlg_data/": "sh40:s3://AVP_BEV/mnt/lustre/share_data/zhongligeng/data/xzlg_data/",
        "s3://AVP_BEV/2023_03_baidu_short/": "sh41:s3://AVP_BEV/2023_03_baidu_short/",
        "s3://sdc_gt_label/": "adsdc:s3://sdc_gt_label/",
        "s3://sdc_adas_3/": "adsdc:s3://sdc_adas_3/",
        "s3://sdc3_gt_label/": "sdc3_gt:s3://sdc3_gt_label/",
        "s3://3DGOP_A02_MM11V/": "aoss:s3://3DGOP_A02_MM11V/",
            }))
file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.5, 0.5),
        "final_dim": (540, 960),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 1080,
        "W": 1920,
        "rand_flip": False,
    }

# resize_w, resize_h, crop_x_left, crop_y_top, crop_width, crop_height 
resize_crop_info = {
    'center_camera_fov20': [1536, 864, 256, (864-512)//2, 1024, 512],
    'center_camera_fov60': [1024, 576, 0, (576-512)//2, 1024, 512], #fov30
    'center_camera_fov105': [1024, 576, 0, (576-512)//2, 1024, 512], #fov105, mono
}

resize_crop_info_bev = {
    'center_camera_fov30': [1024, 576, 0, (576-512)//2, 1024, 512], #fov30
    'center_camera_fov120': [1024, 576, 0, (576-512)//2, 1024, 512], #[1820, 1024, 398, 256, 1024, 512], #fov60
    'left_front_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'left_rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'right_rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'right_front_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'center_camera_fov30_1': [1536, 864, 256, (864-512)//2, 1024, 512], #fov20
}

# 实车T68数据，resize_w, resize_h, crop_x_left, crop_y_top, crop_width, crop_height 
resize_crop_info_T68_real = {
    'center_camera_fov20': [1024, 512, 0, 0, 1024, 512],
    'center_camera_fov60': [1024, 512, 0, 0, 1024, 512], #fov30
    'center_camera_fov105': [1024, 512, 0, 0, 1024, 512], #fov105, mono
}

resize_crop_info_bev_T68_real = {
    'center_camera_fov30': [1024, 512, 0, 0, 1024, 512], #fov30
    'center_camera_fov120': [1024, 512, 0, 0, 1024, 512],
    'left_front_camera': [1024, 512, 0, 0, 1024, 512],
    'left_rear_camera': [1024, 512, 0, 0, 1024, 512],
    'rear_camera': [1024, 512, 0, 0, 1024, 512],
    'right_rear_camera': [1024, 512, 0, 0, 1024, 512],
    'right_front_camera': [1024, 512, 0, 0, 1024, 512],
    'center_camera_fov20': [1024, 512, 0, 0, 1024, 512], #fov20
}

reprojection_keys = ['gt_bboxes_2d', 'gt_2d_to_3d_idx', 'gt_labels_2d', 'gt_cam_idx']
gop_valid_keys=['fuse_cls_weights','is_gop_data']
collect_keys=collect_keys

train_pipeline_T68_real = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=512, output_width=1024, use_virtual_cam=False, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=512, output_width=1024, use_virtual_cam=False, cams=cams_pinhole_crop_map),  ###
    dict(type='LoadAnnotations2D3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True, with_bbox_depth=True, with_mono_rpn=True, 
                                     with_bbox_crop=True, with_label_crop=True, with_bboxes_depth_crop=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ObjectNameFilter2D3D', classes=class_names),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_bev_T68_real, training=True),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info_T68_real, training=True),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='SPETRFormatBundlePinholeFisheye', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_pinhole', 'img_pinhole_crop', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'gt_bboxes_3d_cam',
               'gt_bboxes_crop', 'gt_labels_crop', 'gt_bboxes_3d_cam_crop', 'centers2d_crop', 'depths_crop'] + collect_keys + reprojection_keys,
         meta_keys=('filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole', 'extrinsics_pinhole',
                    'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', 'lidar2img_pinhole_crop',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d', 'depth_factors_crop', 'depth_factors_PF'))
]

train_pipeline_ori = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=576, output_width=1024, use_virtual_cam=False, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=2160, output_width=3840, use_virtual_cam=False, cams=cams_pinhole_crop_map),  ###
    dict(type='LoadAnnotations2D3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True, with_bbox_depth=True, with_mono_rpn=True, 
                                     with_bbox_crop=True, with_label_crop=True, with_bboxes_depth_crop=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ObjectNameFilter2D3D', classes=class_names),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_bev, training=True),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info, training=True),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='SPETRFormatBundlePinholeFisheye', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_pinhole', 'img_pinhole_crop', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'gt_bboxes_3d_cam',
               'gt_bboxes_crop', 'gt_labels_crop', 'gt_bboxes_3d_cam_crop', 'centers2d_crop', 'depths_crop'] + collect_keys + reprojection_keys,
         meta_keys=('filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole', 'extrinsics_pinhole',
                    'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', 'lidar2img_pinhole_crop',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d', 'depth_factors_crop', 'depth_factors_PF'))
]

train_pipeline = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=576, output_width=1024, use_virtual_cam=use_virtual_cam, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=2160, output_width=3840, use_virtual_cam=use_virtual_cam, cams=cams_pinhole_crop_map),  ###
    dict(type='LoadAnnotations2D3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True, with_bbox_depth=True, with_mono_rpn=True, 
                                     with_bbox_crop=True, with_label_crop=True, with_bboxes_depth_crop=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ObjectNameFilter2D3D', classes=class_names),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_bev, training=True),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info, training=True),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='SPETRFormatBundlePinholeFisheye', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_pinhole', 'img_pinhole_crop', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'gt_bboxes_3d_cam',
               'gt_bboxes_crop', 'gt_labels_crop', 'gt_bboxes_3d_cam_crop', 'centers2d_crop', 'depths_crop'] + collect_keys + reprojection_keys,
         meta_keys=('filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole', 'extrinsics_pinhole',
                    'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', 'lidar2img_pinhole_crop',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d', 'depth_factors_crop', 'depth_factors_PF'))
]

train_pipeline_ceph = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args_ceph, output_height=576, output_width=1024, use_virtual_cam=use_virtual_cam, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args_ceph, output_height=2160, output_width=3840, use_virtual_cam=use_virtual_cam, cams=cams_pinhole_crop_map),  ###
    dict(type='LoadAnnotations2D3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True, with_bbox_depth=True, with_mono_rpn=True, 
                                     with_bbox_crop=True, with_label_crop=True, with_bboxes_depth_crop=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ObjectNameFilter2D3D', classes=class_names),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_bev, training=True),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info, training=True),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='SPETRFormatBundlePinholeFisheye', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img_pinhole', 'img_pinhole_crop', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'prev_exists', 'gt_bboxes_3d_cam',
               'gt_bboxes_crop', 'gt_labels_crop', 'gt_bboxes_3d_cam_crop', 'centers2d_crop', 'depths_crop'] + collect_keys + reprojection_keys,
         meta_keys=('filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole', 'extrinsics_pinhole',
                    'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', 'lidar2img_pinhole_crop',
                    'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d', 'depth_factors_crop', 'depth_factors_PF'))
]


resize_crop_info_test = {
    'center_camera_fov105': [1024, 576, 0, (576-512)//2, 1024, 512],
}
resize_crop_info_test_bev = {
    'center_camera_fov30': [1024, 576, 0, (576-512)//2, 1024, 512], #fov30
    'center_camera_fov120': [1024, 576, 0, (576-512)//2, 1024, 512], #[1820, 1024, 398, 256, 1024, 512], #fov60
    'left_front_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'left_rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'right_rear_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'right_front_camera': [1024, 576, 0, (576-512)//2, 1024, 512],
    'center_camera_fov30_1': [1536, 864, 256, (864-512)//2, 1024, 512], #fov20
}

## T68实车数据
resize_crop_info_test_T68_real = {
    'center_camera_fov105': [1024, 512, 0, 0, 1024, 512],
}
resize_crop_info_test_bev_T68_real = {
    'center_camera_fov30': [1024, 512, 0, 0, 1024, 512], #fov30
    'center_camera_fov120': [1024, 512, 0, 0, 1024, 512], #fov120
    'left_front_camera': [1024, 512, 0, 0, 1024, 512],
    'left_rear_camera': [1024, 512, 0, 0, 1024, 512],
    'rear_camera': [1024, 512, 0, 0, 1024, 512],
    'right_rear_camera': [1024, 512, 0, 0, 1024, 512],
    'right_front_camera': [1024, 512, 0, 0, 1024, 512],
    'center_camera_fov20': [1024, 512, 0, 0, 1024, 512], #fov20
}

test_pipeline_T68_real = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=512, output_width=1024, use_virtual_cam=False, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=512, output_width=1024, use_virtual_cam=False, cams=cams_pinhole_crop_map),  ###
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_test_bev_T68_real, training=False),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info_test_T68_real, training=False),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[dict(type='SPETRFormatBundlePinholeFisheye', collect_keys=collect_keys, class_names=class_names, with_label=False),
                     dict(type='Collect3D',
                          keys=['img_pinhole', 'img_pinhole_crop'] + collect_keys,
                          meta_keys=(
                            'filename_fisheye', 'ori_shape_fisheye', 'img_shape_fisheye', 'pad_shape_fisheye', 'scale_factor_fisheye', "ori_intrinsics_fisheye", 'intrinsics_fisheye', 'extrinsics_fisheye', 'intrinsics_pinhole', 'extrinsics_pinhole',
                            'filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole',
                            'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', "ori_intrinsics_pinhole_crop", 'intrinsics_pinhole_crop', 'extrinsics_pinhole_crop', 'lidar2img_pinhole_crop',
                            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'depth_factors_crop', 'depth_factors_PF'))])
]

test_pipeline_ori = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=576, output_width=1024, use_virtual_cam=False, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=2160, output_width=3840, use_virtual_cam=False, cams=cams_pinhole_crop_map_test),  ###
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_test_bev, training=False),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info_test, training=False),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[dict(type='SPETRFormatBundlePinholeFisheye', collect_keys=collect_keys, class_names=class_names, with_label=False),
                     dict(type='Collect3D',
                          keys=['img_pinhole', 'img_pinhole_crop'] + collect_keys,
                          meta_keys=(
                            'filename_fisheye', 'ori_shape_fisheye', 'img_shape_fisheye', 'pad_shape_fisheye', 'scale_factor_fisheye', "ori_intrinsics_fisheye", 'intrinsics_fisheye', 'extrinsics_fisheye', 'intrinsics_pinhole', 'extrinsics_pinhole',
                            'filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole',
                            'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', "ori_intrinsics_pinhole_crop", 'intrinsics_pinhole_crop', 'extrinsics_pinhole_crop', 'lidar2img_pinhole_crop',
                            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'depth_factors_crop', 'depth_factors_PF'))])
]

test_pipeline = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=576, output_width=1024, use_virtual_cam=use_virtual_cam, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args, output_height=2160, output_width=3840, use_virtual_cam=use_virtual_cam, cams=cams_pinhole_crop_map_test),  ###
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_test_bev, training=False),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info_test, training=False),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[dict(type='SPETRFormatBundlePinholeFisheye', collect_keys=collect_keys, class_names=class_names, with_label=False),
                     dict(type='Collect3D',
                          keys=['img_pinhole', 'img_pinhole_crop'] + collect_keys,
                          meta_keys=(
                            'filename_fisheye', 'ori_shape_fisheye', 'img_shape_fisheye', 'pad_shape_fisheye', 'scale_factor_fisheye', "ori_intrinsics_fisheye", 'intrinsics_fisheye', 'extrinsics_fisheye', 'intrinsics_pinhole', 'extrinsics_pinhole',
                            'filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole',
                            'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', "ori_intrinsics_pinhole_crop", 'intrinsics_pinhole_crop', 'extrinsics_pinhole_crop', 'lidar2img_pinhole_crop',
                            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'depth_factors_crop', 'depth_factors_PF'))])
]

test_pipeline_ceph = [
    dict(type='LoadMultiBEVImageFromFiles', to_float32=True, file_client_args=file_client_args_ceph, output_height=576, output_width=1024, use_virtual_cam=use_virtual_cam, cams=cams_pinhole),  ###
    dict(type='LoadMultiMonoImageFromFiles', to_float32=True, file_client_args=file_client_args_ceph, output_height=2160, output_width=3840, use_virtual_cam=use_virtual_cam, cams=cams_pinhole_crop_map_test),  ###
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range_mask2d3d),
    dict(type='ResizeCropFlipRotBEVImage', resize_crop_info=resize_crop_info_test_bev, training=False),
    dict(type='ResizeCropFlipRotMonoImage', resize_crop_info=resize_crop_info_test, training=False),
    dict(type='NormalizeMultiPinholeFisheyeImage', **img_norm_cfg, img_types=['img_pinhole', 'img_pinhole_crop']),
    dict(type='PadMultiPinholeFisheyeImage', size_divisor=32),
    dict(type='MultiScaleFlipAug3D',
         img_scale=(1333, 800),
         pts_scale_ratio=1,
         flip=False,
         transforms=[dict(type='SPETRFormatBundlePinholeFisheye', collect_keys=collect_keys, class_names=class_names, with_label=False),
                     dict(type='Collect3D',
                          keys=['img_pinhole', 'img_pinhole_crop'] + collect_keys,
                          meta_keys=(
                            'filename_fisheye', 'ori_shape_fisheye', 'img_shape_fisheye', 'pad_shape_fisheye', 'scale_factor_fisheye', "ori_intrinsics_fisheye", 'intrinsics_fisheye', 'extrinsics_fisheye', 'intrinsics_pinhole', 'extrinsics_pinhole',
                            'filename_pinhole', 'ori_shape_pinhole', 'img_shape_pinhole', 'pad_shape_pinhole', 'scale_factor_pinhole', 'lidar2img_pinhole',
                            'filename_pinhole_crop', 'ori_shape_pinhole_crop', 'img_shape_pinhole_crop', 'pad_shape_pinhole_crop', 'scale_factor_pinhole_crop', "ori_intrinsics_pinhole_crop", 'intrinsics_pinhole_crop', 'extrinsics_pinhole_crop', 'lidar2img_pinhole_crop',
                            'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token', 'depth_factors_crop', 'depth_factors_PF'))])
]

dataset_type = 'GacDataset'
data_root = '/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/'
ann_file_client_args_ceph = dict(backend='petrel', conf_path=conf_path)
ann_file_client_args = dict(backend='disk')


data_11v_gop_local = [
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,  ###
        start_frame=150,
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/train_infos_unknown_3DGOP_A02_MM11V_v1.3_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/train_infos_unknown_3DGOP_A02_MM11V_v1.3_flag.json',
        split_json=True,  ###
        num_frame_losses=num_frame_losses,
        seq_split_num=1, # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,  ###
        start_frame=150,
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V_QT/jsons/V1.0/infos_unknown_3DGOP_A02_MM11V_QT_v1.0_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V_QT/jsons/V1.0/infos_unknown_3DGOP_A02_MM11V_QT_v1.0_flag.json',
        split_json=True,  ###
        num_frame_losses=num_frame_losses,
        seq_split_num=1, # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline,
        classes=class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-292',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,  ###
        start_frame=150,
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02/jsons/V1.9_part2/train_infos_unknown_3DGOP_A02_v1.9_7Vblocked_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02/jsons/V1.9_part2/train_infos_unknown_3DGOP_A02_v1.9_7Vblocked_flag.json',
        split_json=True,  ###
        num_frame_losses=num_frame_losses,
        seq_split_num=1, # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ori,
        classes=class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=False,
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,  ###
        start_frame=0,
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_0/train_infos_3DGOP_T68_batch_0_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_0/train_infos_3DGOP_T68_batch_0_flag.json',
        split_json=True,  ###
        num_frame_losses=num_frame_losses,
        seq_split_num=1, # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ori,
        classes=class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=False,
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,  ###
        start_frame=0,
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_1/train_infos_3DGOP_T68_batch_1_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_1/train_infos_3DGOP_T68_batch_1_flag.json',
        split_json=True,  ###
        num_frame_losses=num_frame_losses,
        seq_split_num=1, # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ori,
        classes=class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=False,
    ),
]

data_11v_test_local = dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/val_infos_unknown_3DGOP_A02_MM11V_v1.3_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/val_infos_unknown_3DGOP_A02_MM11V_v1.3_flag.json',
        split_json=True,
        pipeline=test_pipeline,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map_test,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        sample_rate=1,  ###
        start_frame=150,
        seq_mode=True,   # streaming video training
        evaluator_type='PRD',
        cls_conf_threshold=0.2,
        cls_iou_threshold=0.5,
        point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',  ###
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
        )

data_11v_quant_A02_city = dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68_7V/jsons_A02_daytime_dark/A02_city_daytime_dark.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68_7V/jsons_A02_daytime_dark/A02_city_daytime_dark_flag.json',
        split_json=True,
        pipeline=test_pipeline,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map_test,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        sample_rate=1,  ###
        start_frame=0,
        seq_mode=True,   # streaming video training
        evaluator_type='PRD',
        cls_conf_threshold=0.1,
        cls_iou_threshold=0.5,
        point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',  ###
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
        )

data_11v_test_ext = dict(
    type=dataset_type,
    data_root=data_root,
    file_client_args=ann_file_client_args_ceph,
    ann_file='sdcoss2_lsm:s3://sdcoss2_lsm/PAP/gac_a02_11v_auotlabel_20240410_768_23413_cepth_paths_spetr_len10548_seq74_min100fr_max199fr_tmax0.25s.json',
    flag_file='sdcoss2_lsm:s3://sdcoss2_lsm/PAP/gac_a02_11v_auotlabel_20240410_768_23413_cepth_paths_spetr_len10548_seq74_min100fr_max199fr_tmax0.25s_flag.json',
    split_json=True,
    pipeline=test_pipeline_ceph,
    classes=class_names,
    # sub_class_names=sub_class_names,
    cams=cams,
    cams_fisheye=cams_fisheye,
    cams_pinhole=cams_pinhole,
    cams_pinhole_crop_map=cams_pinhole_crop_map_test,
    modality=input_modality,
    collect_keys=collect_keys + \
    ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
    queue_length=queue_length,
    test_mode=True,
    sample_rate=1,
    seq_mode=True,   # streaming video training
    evaluator_type='PRD',
    cls_conf_threshold=0.3,
    cls_iou_threshold=0.5,
    point_cloud_range=point_cloud_range,
    extrinsic_adjust_camera_name=None,
    extrinsic_adjust_camera2centerprime=None,
    box_type_3d='LiDAR',
    with_2d_label=False,
    object_in_lidar_coord=False,
    json_fisheye_model='Ocam',
    use_virtual_cam=use_virtual_cam,
    car_type='A02-290',
)

data_11v_gop = [
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='aoss_px:s3://aoss-px1/datasets/a02_3dgop_11v_sensebee_20240531/KB/a02_3dgop_11v_sensebee_20240531_ceph_paths_spetr_len1000_seq5_min200fr_max200fr_tmax0.25s.json',
        flag_file='aoss_px:s3://aoss-px1/datasets/a02_3dgop_11v_sensebee_20240531/KB/a02_3dgop_11v_sensebee_20240531_ceph_paths_spetr_len1000_seq5_min200fr_max200fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='KB',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240612/KB2/gt_gop_deliver_0612_ceph_paths_spetr_len1869_seq14_min15fr_max200fr_tmax0.25s.json',
        flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240612/KB2/gt_gop_deliver_0612_ceph_paths_spetr_len1869_seq14_min15fr_max200fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='KB',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240617/KB2/gt_gop_deliver_20240617_ceph_paths_spetr_len5478_seq44_min11fr_max200fr_tmax0.25s.json',
        flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240617/KB2/gt_gop_deliver_20240617_ceph_paths_spetr_len5478_seq44_min11fr_max200fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='KB',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240620/KB2/gt_gop_deliver_20240620_ceph_paths_spetr_len8084_seq130_min10fr_max200fr_tmax0.25s.json',
        flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240620/KB2/gt_gop_deliver_20240620_ceph_paths_spetr_len8084_seq130_min10fr_max200fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='KB',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240624/oCam2/gt_gop_deliver_20240624_ceph_paths_spetr_len25677_seq174_min10fr_max301fr_tmax0.25s.json',
        flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240624/oCam2/gt_gop_deliver_20240624_ceph_paths_spetr_len25677_seq174_min10fr_max301fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
    ),
    # dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     sample_rate=1,
    #     file_client_args=ann_file_client_args_ceph,
    #     ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240627/oCam2/gt_gop_deliver_20240627_ceph_paths_spetr_len20654_seq236_min10fr_max301fr_tmax0.25s.json',
    #     flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240627/oCam2/gt_gop_deliver_20240627_ceph_paths_spetr_len20654_seq236_min10fr_max301fr_tmax0.25s_flag.json',
    #     split_json=True,
    #     # sub_class_names=sub_class_names,
    #     num_frame_losses=num_frame_losses,
    #     seq_split_num=1,  # streaming video training
    #     seq_mode=True,   # streaming video training
    #     pipeline=train_pipeline_ceph,
    #     classes=class_names,
    #     cams=cams,
    #     cams_fisheye=cams_fisheye,
    #     cams_pinhole=cams_pinhole,
    #     cams_pinhole_crop_map=cams_pinhole_crop_map,
    #     modality=input_modality,
    #     collect_keys=collect_keys + \
    #     ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
    #     queue_length=queue_length,
    #     test_mode=False,
    #     filter_empty_gt=False,
    #     extrinsic_adjust_camera_name=None,
    #     extrinsic_adjust_camera2centerprime=None,
    #     box_type_3d='LiDAR',
    #     with_2d_label=False,
    #     object_in_lidar_coord=False,
    #     json_fisheye_model='Ocam',
    #     use_virtual_cam=use_virtual_cam,
    #     car_type='A02-028',
    # ), dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     sample_rate=1,
    #     file_client_args=ann_file_client_args_ceph,
    #     ann_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240628/oCam2/gt_gop_deliver_20240628_ceph_paths_spetr_len20990_seq267_min10fr_max301fr_tmax0.25s.json',
    #     flag_file='aoss_px:s3://aoss-px1/datasets/gt_gop_deliver_20240628/oCam2/gt_gop_deliver_20240628_ceph_paths_spetr_len20990_seq267_min10fr_max301fr_tmax0.25s_flag.json',
    #     split_json=True,
    #     # sub_class_names=sub_class_names,
    #     num_frame_losses=num_frame_losses,
    #     seq_split_num=1,  # streaming video training
    #     seq_mode=True,   # streaming video training
    #     pipeline=train_pipeline_ceph,
    #     classes=class_names,
    #     cams=cams,
    #     cams_fisheye=cams_fisheye,
    #     cams_pinhole=cams_pinhole,
    #     cams_pinhole_crop_map=cams_pinhole_crop_map,
    #     modality=input_modality,
    #     collect_keys=collect_keys + \
    #     ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
    #     queue_length=queue_length,
    #     test_mode=False,
    #     filter_empty_gt=False,
    #     extrinsic_adjust_camera_name=None,
    #     extrinsic_adjust_camera2centerprime=None,
    #     box_type_3d='LiDAR',
    #     with_2d_label=False,
    #     object_in_lidar_coord=False,
    #     json_fisheye_model='Ocam',
    #     use_virtual_cam=use_virtual_cam,
    #     car_type='A02-028',
    # )
]

data_11v_gop_pvb_fuse = [
    dict(
        type=dataset_type,
        data_root=data_root,
        sample_rate=1,
        file_client_args=ann_file_client_args_ceph,
        ann_file='sdc2:s3://px_data/gop_gts/20240603_case60_new/gac_a02_11v_gop_20240603_case60_ceph_paths_ceph_paths_spetr_len75494_seq255_min205fr_max301fr_tmax0.25s.json',
        flag_file='sdc2:s3://px_data/gop_gts/20240603_case60_new/gac_a02_11v_gop_20240603_case60_ceph_paths_ceph_paths_spetr_len75494_seq255_min205fr_max301fr_tmax0.25s_flag.json',
        split_json=True,
        # sub_class_names=sub_class_names,
        num_frame_losses=num_frame_losses,
        seq_split_num=1,  # streaming video training
        seq_mode=True,   # streaming video training
        pipeline=train_pipeline_ceph,
        classes=class_names,
        cams=cams,
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + \
        ['img_pinhole', 'img_pinhole_crop', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        filter_empty_gt=False,
        extrinsic_adjust_camera_name=None,
        extrinsic_adjust_camera2centerprime=None,
        box_type_3d='LiDAR',
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='KB',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-934',
    )
]

data_11v_test_A02 = dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68_7V/jsons_A02/V0.0/case_infos_A02_PVBGOP_v0.0.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68_7V/jsons_A02/V0.0/case_infos_A02_PVBGOP_v0.0_flag.json',
        split_json=True,
        pipeline=test_pipeline,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map_test,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        sample_rate=1,  ###
        start_frame=0,
        seq_mode=True,   # streaming video training
        evaluator_type='PRD',
        cls_conf_threshold=0.2,
        cls_iou_threshold=0.5,
        point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',  ###
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=use_virtual_cam,
        car_type='A02-459',
        )

data_11v_batch_case1_T68 = dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustre/xuzhiyong/code/detr3d_gact68/tmp_json/T68_batch_case1_gop_2024_09_02_17_13_57_AutoCollect.json',
        flag_file='/mnt/lustre/xuzhiyong/code/detr3d_gact68/tmp_json/T68_batch_case1_gop_2024_09_02_17_13_57_AutoCollect_flag.json',
        split_json=True,
        pipeline=test_pipeline_T68_real,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map_test,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        sample_rate=1,  ###
        seq_mode=True,   # streaming video training
        evaluator_type='PRD',
        cls_conf_threshold=0.1,
        cls_iou_threshold=0.5,
        point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',  ###
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=False
        )

data_11v_quant_batch_case4_600frames_T68_real = dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_case4/T68_batch_case4_quant.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_T68/jsons/batch_case4/T68_batch_case4_flag_quant.json',
        split_json=True,
        pipeline=test_pipeline_T68_real,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map_test,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        sample_rate=1,  ###
        seq_mode=True,   # streaming video training
        evaluator_type='PRD',
        cls_conf_threshold=0.1,
        cls_iou_threshold=0.5,
        point_cloud_range=point_cloud_range,
        extrinsic_adjust_camera_name=None,  ###
        extrinsic_adjust_camera2centerprime=None,  ###
        box_type_3d='LiDAR',  ###
        with_2d_label=False,
        object_in_lidar_coord=False,
        json_fisheye_model='Ocam',
        use_virtual_cam=False
        )

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    train=data_11v_gop_local,
    val=dict(
        type=dataset_type,
        data_root=data_root,  ###
        file_client_args=ann_file_client_args,  ###
        ann_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/val_infos_unknown_3DGOP_A02_MM11V_v1.3_sh36.json',
        flag_file='/mnt/lustrenew/share_data/xuzhiyong/3DGOP_A02_MM11V/jsons/V1.3/val_infos_unknown_3DGOP_A02_MM11V_v1.3_flag.json',
        split_json=True,
        pipeline=test_pipeline,
        classes=class_names,
        # sub_class_names=sub_class_names,
        cams=cams,  ###
        cams_fisheye=cams_fisheye,
        cams_pinhole=cams_pinhole,
        cams_pinhole_crop_map=cams_pinhole_crop_map,
        modality=input_modality,
        collect_keys=collect_keys + ['img_pinhole', 'img_pinhole_crop', 'img_metas'],
        queue_length=queue_length,
        test_mode=True,  ###
        extrinsic_adjust_camera_name="center_camera_fov120",  ###
        extrinsic_adjust_camera2centerprime=fov120_2_center_prime,  ###
        box_type_3d='LiDAR',  ###
        use_virtual_cam=use_virtual_cam,
        car_type='A02-290',
        ),
    # test=data_11v_test_local,  # A02虚拟变换测试集
    # test=data_11v_test_A02,  # A02虚拟变换，一望无际锥桶
    # test=data_11v_quant_A02_city,  # A02虚拟相机变换，密集城区、夜间
    # test=data_11v_batch_case1_T68,  # T68左侧一排锥桶
    test=data_11v_quant_batch_case4_600frames_T68_real,  # 真实T68路测数据，1024×512, V0.3 batch_case4挑选600帧量化数据集
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(
    type='AdamW', 
    lr=data['samples_per_gpu'] / 2.0 / 4 * 2e-4,
    paramwise_cfg=dict(
        custom_keys={
            # 'img_backbone': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
            'img_backbone_pinhole': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
            'img_backbone_fisheye': dict(lr_mult=0.25), # 0.25 only for Focal-PETR with R50-in1k pretrained weights
        }),
    weight_decay=0.01)

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic', grad_clip=dict(max_norm=105, norm_type=2))  ###
# optimizer_config = dict(grad_clip=dict(max_norm=105, norm_type=2))  ###

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

# evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)
evaluation = dict(interval=100000000, pipeline=test_pipeline, iou_2d_threshold=[0.5]*len(class_names))  ###
find_unused_parameters = True  # when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=50)
runner = dict(type='Custom_IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
# runner = dict(type='EpochBasedRunner', max_epochs=num_epochs)  ###
freeze_module = dict(
    fix_modules=['img_backbone_pinhole']
)
load_from = '/mnt/lustrenew/share/xuzhiyong/T68_code/V0.7_pretrain_iter_156780.pth'
resume_from = None

# adela deployment config
import json
rpn_deploy_config = json.dumps({
  "horizon": {
    "import": {
      "model": {
        "remove_node_type": "Dequantize;Quantize",
        "march": "nash-m"
      },
      "calibration": {
        "optimization": "set_all_nodes_int16;set_Softmax_input_int16;set_MatMul_input_int8;bias_correction",
        "calibration_type": "mix"
      }
    }
  },
  "dump_info": True,
  "__nart_trace__": True
})
head_deploy_config = json.dumps({
  "horizon": {
    "import": {
      "model": {
        "remove_node_type": "Dequantize;Quantize",
        "march": "nash-m",
        "node_info": {
          "MatMul_319": {
            "InputType0": "int16"
          },
          "MatMul_768": {
            "InputType1": "int16"
          },
          "MatMul_770": {
            "InputType0": "int16"
          },
          "MatMul_839": {
            "InputType1": "int16"
          },
          "MatMul_841": {
            "InputType0": "int16"
          },
          "MatMul_917": {
            "InputType1": "int16"
          },
          "MatMul_919": {
            "InputType0": "int16"
          },
          "MatMul_988": {
            "InputType1": "int16"
          },
          "MatMul_990": {
            "InputType0": "int16"
          },
          "MatMul_1066": {
            "InputType1": "int16"
          },
          "MatMul_1068": {
            "InputType0": "int16"
          },
          "MatMul_1137": {
            "InputType1": "int16"
          },
          "MatMul_1139": {
            "InputType0": "int16"
          }
        }
      },
      "calibration": {
        "optimization": "set_all_nodes_int16;set_Softmax_input_int16;set_MatMul_input_int8;bias_correction",
        "calibration_type": "mix"
      }
    }
  },
  "dump_info": True,
  "__nart_trace__": True
})
deploy_config = json.dumps({
  "horizon": {
    "import": {
      "model": {
        "remove_node_type": "Dequantize;Quantize",
        "march": "nash-m"
            }
        },
    "preprocess": {
      "center_camera_fov30": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "center_camera_fov120": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "rear_camera": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "left_front_camera": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "left_rear_camera": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "right_front_camera": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "right_rear_camera": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "center_camera_fov20": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      },
      "center_camera_fov105": {
        "means": [
          123.675,
          116.28,
          103.53
        ],
        "stds": [
          58.395,
          57.12,
          57.375
        ],
        "color_type_ori": "rgb",
        "color_type_rt": "nv12"
      }
    }
  },
  "dump_info": True,
  "__nart_trace__": True
})

deployment_cfg = dict(
    project_id=20,
    model_name='spetr_8p_pvbgop',
    input_h_pinhole_crop=512,
    input_w_pinhole_crop=1024,
    input_h_pinhole=512,
    input_w_pinhole=1024,
    output_names_backbone=['img_feats_pinhole'],
    output_names_backbone_pinhole=['img_feats_pinhole'],
    output_names_head=["score", "bbox", "embedding", "pseudo_reference_points"],
    deployment_platforms=["acl-ascend615-fp16-adc615-sdk230808"],
    deploy_configs=[deploy_config],
    head_deploy_configs=[head_deploy_config],
    rpn_deploy_configs=[rpn_deploy_config]
)
