# dataset settings
dataset_type = 'FlairOneDataset'
data_root = '/mmsegmentation/data/flair/'
crop_size = (512, 512)
img_channels = 5

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

	
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile', in_channels=img_channels),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(  
    batch_size=64,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='InfiniteSampler', shuffle=True),  
    dataset=dict( 
        type=dataset_type,  
        data_root=data_root,  
        data_prefix=dict(
            img_path='train/images', seg_map_path='train/masks'),  
        pipeline=train_pipeline)) 

val_dataloader = dict(
    batch_size=64,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False),  
    dataset=dict( 
        type=dataset_type,  
        data_root=data_root,  
        data_prefix=dict(
            img_path='val/images', seg_map_path='val/masks'),  
        pipeline=test_pipeline))  

test_dataloader = dict(
    batch_size=64,  
    num_workers=10,  
    persistent_workers=True,  
    sampler=dict(type='DefaultSampler', shuffle=False),  
    dataset=dict( 
        type=dataset_type,  
        data_root=data_root,  
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/masks'),  
        pipeline=test_pipeline))  

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
