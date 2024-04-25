# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class StdlSoilsDataset(BaseSegDataset):
    """StdlSoils dataset.
    """

    METAINFO = dict(        

        classes=('batiment', 'surface_non_beton', 'surface_beton', 'roche_dure_meuble', 'eau_naturelle', 'roseliere',
               'sol_neige', 'sol_vegetalise', 'sol_divers', 'sol_vigne', 'sol_agricole', 'sol_bache'),
        
        palette=[[219,14,154],[147,142,123],[248,12,0],[169,113,1],[21,83,174],[25,74,38],[70,228,131],
                 [243,166,13],[102,0,130],[85,255,0],[255,243,13],[228,223,124]],
        
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
            12: 11
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

