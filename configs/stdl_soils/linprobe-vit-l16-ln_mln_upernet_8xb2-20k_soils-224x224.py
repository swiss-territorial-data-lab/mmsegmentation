_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/stdl_soils.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (224, 224)

data_preprocessor = dict(  
    type='SegDataPreProcessor',  
    mean=[96.486855, 103.2926, 85.37917, 115.856064, 22.425655],  
    std=[34.910034, 33.61709, 32.622425, 34.69036, 19.819977],  
    bgr_to_rgb=False,  
    rgb_to_bgr=False,  
    size=(224, 224),
    pad_val=0,  
    seg_pad_val=255) 

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='VisionTransformer',
        img_size=crop_size,
        patch_size=16,
        embed_dims=1024,
        in_channels=5,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,  
        frozen_exclude=[],
        # final_norm=True,
        out_indices=[7, 11, 15, 23],
        init_cfg = dict(type='Pretrained', checkpoint='/mnt/Data2/sli/mmsegmentation/pretrained_ViT/vit_sampled_latest.pth')
        ),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=1024,
        scales=[4, 2, 1, 0.5]),   
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=15000),
        in_channels=[1024, 1024, 1024, 1024], 
        num_classes=13, 
        channels=1024,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0)),
    auxiliary_head=dict(
        in_channels=1024, 
        num_classes=13,
        loss_decode=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=0.4)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(144, 144)))
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=2e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(  
    batch_size=16,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='InfiniteSampler', shuffle=True)) 

val_dataloader = dict(
    batch_size=64,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False))  

test_dataloader = dict(
    batch_size=64,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False))  


train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=20000, val_interval=2000)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
