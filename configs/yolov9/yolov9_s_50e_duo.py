_base_ = [
    '../_base_/datasets/duo_detection_mmyolo.py',
    '../_base_/default_runtime_mmyolo.py',
]
env_cfg = dict(cudnn_benchmark=True)

max_epochs = 50
num_last_epochs = 10
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=10, dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

strides = (8, 16, 32)
train_use_auxiliary = True
test_use_auxiliary = False

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    # -----inf-----
    # score_thr=0.25,
    # nms=dict(type='nms', iou_threshold=0.45),
    # max_per_img=1000,
    # -----val-----
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    # nms=dict(type="nms", iou_threshold=0.7, class_agnostic=True),
    max_per_img=300,
)  # Max number of detections of each image

# anchors for DUO
anchors = [[(13, 12), (20, 18), (27, 25)],
           [(35, 31), (44, 39), (55, 52)],
           [(80, 45), (74, 69), (116, 102)]]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=_base_.train_bs),
    constructor='YOLOv7OptimWrapperConstructor')
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=0.01,
        max_epochs=max_epochs),
)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

num_classes = 4

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
backbone=dict(
        type='YOLOv9Backbone',
        arch='s',
        input_channels=3,
        stem_channels=32,
        out_indices=(2, 3, 4),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='s',
        in_channels=[128, 192, 256],
        out_channels=[128, 192, 256],
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
        norm_cfg= dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            num_classes=num_classes,
            in_channels=[128, 192, 256],
            aux_in_channels=[128, 192, 256],
            reg_max=16,
            reg_feat_channels=64,
            cls_feat_channels=128,
            aux_reg_feat_channels=64,
            aux_cls_feat_channels=128,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
            featmap_strides=strides,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
        ),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight,
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False,
        ),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight,
        ),
        aux_weight=0.25,
    ),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9,
        )),
    test_cfg=model_test_cfg,
)
