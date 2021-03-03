# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np

from bottomdetection.annotation import Annotation


def to_bottom_annotation(zarr_data, bottom_depths):
    """
    Creates annotation with a mask for the bottom.
    """

    ping_time = zarr_data['ping_time']
    channel_ids = zarr_data['channelID'].data

    annotation = Annotation()

    for i, bottom_depth in enumerate(bottom_depths):
        if np.isnan(bottom_depth):
            continue
        time = np.datetime64(ping_time[i].data)
        for channel_id in channel_ids:
            annotation.append(
                pingTime=time,
                mask_depth_upper=bottom_depth,
                mask_depth_lower=9999,
                priority=2,
                acousticCat=999,
                proportion=1,
                ID='bottom',
                ChannelID=channel_id,
            )

    return annotation
