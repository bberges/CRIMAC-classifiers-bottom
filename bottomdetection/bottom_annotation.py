# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np

from bottomdetection.annotation import AnnotationInfo, AnnotationMask, Annotation


def to_bottom_annotation(zarr_data, bottom_depths):
    """
    Creates annotation with a mask for the bottom.
    """

    epoch = np.datetime64('1970-01-01T00:00:00')
    one_nanosecond = np.timedelta64(1, 'ns')
    nanoseconds_from_1601_to_1970 = (epoch - np.datetime64('1601-01-01T00:00:00')) / np.timedelta64(1, 's') * 1e9

    def to_nanoseconds_since_1601(time):
        return (time - epoch) / one_nanosecond + nanoseconds_from_1601_to_1970

    ping_time = np.array([to_nanoseconds_since_1601(time) for time in zarr_data.ping_time])
    frequency = zarr_data.frequency
    channel_count = len(frequency)
    channels = zarr_data.channelID.data.tolist()

    no_bottom_indexes = np.argwhere(np.isnan(bottom_depths))
    mask_bottom_depths = np.delete(bottom_depths, no_bottom_indexes)
    mask_ping_time = np.delete(ping_time, no_bottom_indexes)

    bottom_category_id = 999
    bottom_end_depth = 9999

    if len(mask_ping_time) > 0:
        bottom_mask = AnnotationMask(
            region_id='1',
            region_name='bottom',
            region_provenance='CRIMAC',
            region_type='marker',
            region_channels=channels,
            region_category_ids=np.repeat([[bottom_category_id]], channel_count, axis=0),
            region_category_proportions=np.repeat([[1]], channel_count, axis=0),
            start_time=mask_ping_time[0],
            end_time=mask_ping_time[-1],
            min_depth=np.min(mask_bottom_depths),
            max_depth=bottom_end_depth,
            mask_times=mask_ping_time,
            priority=2,
            mask_depth=[[d, bottom_end_depth] for d in mask_bottom_depths])
        bottom_masks = [bottom_mask]
    else:
        bottom_masks = []

    return Annotation(AnnotationInfo(channels, ping_time), bottom_masks)
