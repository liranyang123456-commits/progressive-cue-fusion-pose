_base_ = './ds_tless.py'

train_unseen=True

model = dict(
    type='SCFlow2Refiner',
    cxt_channels=384,
    h_channels=128,
    seperate_encoder=False,
    cxt_feat_detach=True,
    max_flow=400.,
    solve_type='reg',
    add_dense_fusion=True,
    filter_invalid_flow=True,
    encoder=dict(
        type='DINOv2Encoder',
        in_channels=3,
        out_channels=256,
        net_type='basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='SCFlow2Decoder',
        in_channels=3,
        out_channels=256,
        net_type='basic',
        norm_cfg=dict(type='BN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='SCFlow2Decoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=8,
        cxt_channels=384,
        detach_flow=True,
        detach_mask=True,
        detach_pose=True,
        detach_depth_for_xy=True,
        depth_based_upsample=False,
        mask_flow=False,
        mask_corr=False,
        pose_head_cfg=dict(
            type='SceneFlowPoseHead',
            in_channels=16,
            net_type='Basic',
            rotation_mode='ortho6d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
        ),
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')),
    freeze_bn=False,     # False
    freeze_encoder=False,
    train_cfg=dict(
        rendered_mask_filte=True,
        online_image_renderer=train_unseen,         # add in 240726
    ),
    test_cfg=dict(
        iters=8,
    ),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint='work_dirs/dgflow_unseen/scflow2_rawbbox/dinov2b_rawbbox_iter_145000.pth'
    # )
)


interval = 5000
optimizer_config = dict(grad_clip=dict(max_norm=10.))   # 1/3/5     default = 10
steps = 200000
optimizer = dict(
    type='Adam',
    lr=0.0001,
    betas=(0.5, 0.999),
    eps=1e-06,
    weight_decay=0.0,
    amsgrad=False,
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup_ratio=0.001,
    warmup_iters=1000,
    warmup='linear',
    min_lr=0,
)

evaluation=dict(interval=interval,         # evaluation
                metric={
                    # 'bop':[],
                    'auc':[],
                    'add':[0.05, 0.10, 0.20, 0.50]},
                save_best='average/add_10',
                rule='greater'
            )
runner = dict(type='IterBasedRunner', max_iters=steps)
num_gpus = 1
checkpoint_config = dict(interval=interval, by_epoch=False)
log_config=dict(interval=50,
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardImgLoggerHook', interval=50, image_format='HWC')])

work_dir = 'work_dirs/debug'
