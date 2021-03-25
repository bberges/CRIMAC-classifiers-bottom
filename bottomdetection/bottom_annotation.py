# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import pandas as pd
import xarray as xr


def to_bottom_annotation(channel_id: str, bottom_depths: xr.DataArray) -> pd.DataFrame:
    """
    Creates annotation with a mask for the bottom.
    """

    df = bottom_depths.to_dataframe('mask_depth_upper')

    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'ping_time': 'pingTime'})

    df = df.assign(mask_depth_lower=9999,
                   priority=2,
                   acousticCat=999,
                   proportion=1,
                   ID='bottom',
                   ChannelID=channel_id)

    return df
