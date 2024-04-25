# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class FlairOneDataset(BaseSegDataset):
    """FlairOne dataset.
    """

    METAINFO = dict(
        # all 19 classes
        # classes=('Building', 'Pervious surface', 'Impervious surface', 'Bare soil', 'Water', 'Coniferous',
        #        'Deciduous', 'Brushwood', 'Vineyard', 'Herbaceous vegetation', 'Agricultural land', 'Plowed land',
        #        'swimming_pool', 'snow', 'clear cut', 'mixed', 'ligneous', 'greenhouse', 'Other'),
        
        # palette=[[219,14,154],[147,142,123],[248,12,0],[169,113,1],[21,83,174],[25,74,38],[70,228,131],
        #          [243,166,13],[102,0,130],[85,255,0],[255,243,13],[228,223,124],[61,230,235],[255,255,255], 
        #          [138,179,160],[107,113,79],[197,220,66],[153,153,255],[0,0,0]],
        
        # reduced 13 classes
        classes=('Building', 'Pervious surface', 'Impervious surface', 'Bare soil', 'Water', 'Coniferous',
               'Deciduous', 'Brushwood', 'Vineyard', 'Herbaceous vegetation', 'Agricultural land', 'Plowed land',
               'Other'),
        
        palette=[[219,14,154],[147,142,123],[248,12,0],[169,113,1],[21,83,174],[25,74,38],[70,228,131],
                 [243,166,13],[102,0,130],[85,255,0],[255,243,13],[228,223,124],[0,0,0]],
        
        # label 255 will be ignored when calculating loss
        label_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7,
            9: 8,
            10: 9,
            11: 10,
            12: 11,
            13: 12,
            14: 12,
            15: 12,
            16: 12,
            17: 12,
            18: 12,
            19: 12
            }
        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

