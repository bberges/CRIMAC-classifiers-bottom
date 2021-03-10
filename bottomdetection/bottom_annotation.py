# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np

from bottomdetection.annotation import Annotation


def to_bottom_annotation(bottom_depths):
    """
    Creates annotation with a mask for the bottom.
    """

    df = bottom_depths.to_dataframe("mask_depth_upper")

    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'ping_time': 'pingTime'})

    df = df.drop(columns=['latitude', 'longitude', 'raw_file', 'frequency'])
    df = df.assign(mask_depth_lower = 9999,
                priority = 2,
                acousticCat = 999,
                proportion = 1,
                ID = 'bottom')

    return df