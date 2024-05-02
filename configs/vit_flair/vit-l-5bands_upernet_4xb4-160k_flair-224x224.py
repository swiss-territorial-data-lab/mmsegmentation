_base_ = './vit-l-5bands_upernet_4xb4-160k_flair-base.py'

crop_size = (224, 224)
img_channels = 5

model = dict(
    backbone=dict(
        img_size=crop_size,
        in_channels=img_channels,
        frozen_exclude=['all'],
        init_cfg = dict(type='Pretrained', checkpoint='')
        ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(144, 144)))


train_pipeline = [
    dict(type='LoadSingleRSImageFromFile', in_channels=img_channels),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True
        ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline)) 