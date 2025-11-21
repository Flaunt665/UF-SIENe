_base_ = [
    '../_base_/datasets/tsc_detection_mmyolo.py',
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
    one2one_withnms=False,
    one2many_withnms=True,
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    max_per_img=300)  # Max number of detections of each image


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

num_classes = 23

deepen_factor = 0.33
widen_factor = 0.5

last_stage_out_channels = 1024

one2many_tal_topk = 10  # Number of bbox selected in each level
one2many_tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
one2many_tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

one2one_tal_topk = 1  # Number of bbox selected in each level
one2one_tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
one2one_tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics

infer_type = "one2one"

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='YOLOv10Backbone',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        use_c2fcib=True,  # Set to False only in YOLOv10 Nano
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv10PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv10Head',
        head_module=dict(
            type='YOLOv10HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        infer_type=infer_type,
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        one2many_assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=one2many_tal_topk,
            alpha=one2many_tal_alpha,
            beta=one2many_tal_beta,
            eps=1e-9),
        one2one_assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=one2one_tal_topk,
            alpha=one2one_tal_alpha,
            beta=one2one_tal_beta,
            eps=1e-9)
    ),
    test_cfg=model_test_cfg
)
