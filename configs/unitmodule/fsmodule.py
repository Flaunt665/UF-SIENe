with_unit_module = True
norm_cfg = dict(type='GN', num_groups=8)
act_cfg = dict(type='ReLU')
c_s1, c_s2 = 32, 32

fs_module = dict(
    type='FSModule',
    unit_backbone=dict(
        type='FSBackbone',
        stem_channels=(c_s1, c_s2),
        embed_dims=(32, 32),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    t_head=dict(
        type='THead',
        in_channels=32,
        hid_channels=32,
        out_channels=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg),
    a_head=dict(type='FSAHead'),
    loss_t=dict(type='TransmissionLoss', loss_weight=500),
    loss_sp=dict(type='SaturatedPixelLoss', loss_weight=0.01),
    loss_tv=dict(type='TotalVariationLoss', loss_weight=0.01),
    loss_cc=dict(type='ColorCastLoss', loss_weight=0.1),
    loss_acc=dict(type='AssistingColorCastLoss', channels=32, loss_weight=0.1),
    alpha=0.9,
    t_min=0.001)
