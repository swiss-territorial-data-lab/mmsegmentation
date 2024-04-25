_base_ = [
    '../_base_/models/upernet_mae.py', '../_base_/datasets/flair_one.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (224, 224)
load_from ='/mnt/Data2/sli/mmsegmentation/pretrained_ViT/pretrained_geneva.pth'

data_preprocessor = dict(  
    type='SegDataPreProcessor',  
    mean=[113.777, 117.952, 109.288, 102.061, 16.763],  
    std=[35.525, 32.141, 30.779, 27.290, 15.896],  
    bgr_to_rgb=False,  
    rgb_to_bgr=False,  
    size=(224, 224),
    pad_val=0,  
    seg_pad_val=255) 

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MAE',
        img_size=crop_size,
        patch_size=16,
        embed_dims=1024,
        in_channels=5,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        # qv_bias=True,
        init_values=1.0,
        drop_path_rate=0.1,
        pretrained=load_from,
        frozen_stages=24,
        out_indices=[7, 11, 15, 23]),
    neck=dict(embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024], num_classes=13, channels=1024),
    auxiliary_head=dict(in_channels=1024, num_classes=13),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(144, 144)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(num_layers=24, decay_rate=0.65),
    constructor='LearningRateDecayOptimizerConstructor')

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

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(  
    batch_size=64,  
    num_workers=1,  
    persistent_workers=True,  
    sampler=dict(type='InfiniteSampler', shuffle=True)) 

val_dataloader = dict(
    batch_size=64,  
    num_workers=1,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False))  

test_dataloader = dict(
    batch_size=64,  
    num_workers=1,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False))  


train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=1000)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
