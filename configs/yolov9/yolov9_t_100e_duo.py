_base_ = [
    '../_base_/datasets/duo_detection_mmyolo.py',
    '../_base_/default_runtime_mmyolo.py',
]
env_cfg = dict(cudnn_benchmark=True)

max_epochs = 100
num_last_epochs = 15
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=10, dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

strides = (8, 16, 32)
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
act_cfg = dict(type='SiLU', inplace=True)
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
    backbone=dict(
        type='YOLOv9Backbone',
        arch='t',
        stem_channels=16,
    ),
    neck=dict(
        type='YOLOv9PAFPN',
        arch='t',
        in_channels=[64, 96, 128],
        out_channels=[64, 96, 128],
        train_use_auxiliary=train_use_auxiliary,
        test_use_auxiliary=test_use_auxiliary,
    ),
    bbox_head=dict(
        type='YOLOv9Head',
        head_module=dict(
            type='YOLOv9HeadModule',
            in_channels=[64, 96, 128],
            aux_in_channels=[64, 96, 128],
            reg_feat_channels=64,
            cls_feat_channels=80,
            aux_reg_feat_channels=64,
            aux_cls_feat_channels=80,
            train_use_auxiliary=train_use_auxiliary,
            test_use_auxiliary=test_use_auxiliary,
        ),
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
