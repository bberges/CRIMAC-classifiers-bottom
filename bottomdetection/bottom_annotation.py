# -*- coding: utf-8 -*-
"""
Conversion of bottom depths to annotation masks.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import pandas as pd
import xarray as xr


def to_pandas(zarr_data: xr.Dataset, channel_index: int, bottom_depth: xr.DataArray) -> pd.DataFrame:
    bottom_depth = bottom_depth.dropna('ping_time')
    channel_id = zarr_data['channel_id'][channel_index].values

    df = bottom_depth.to_dataframe('mask_depth_upper')

    df.reset_index(level=0, inplace=True)

    df = df.assign(mask_depth_lower=9999,
                   priority=2,
                   acoustic_category=999,
                   proportion=1,
                   object_id='bottom',
                   channel_id=channel_id)

    return df


def to_xarray(zarr_data: xr.Dataset, channel_index: int, bottom_depth: xr.DataArray, attributes: dict) -> xr.Dataset:
    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]
    bottom_range = bottom_depth - heave_corrected_transducer_depth

    ds = xr.Dataset(
        data_vars=dict(
            bottom_range=(['ping_time'], bottom_range),
        ),
        coords=dict(
            ping_time=zarr_data['ping_time'],
        ),
    )

    for key in attributes.keys():
        ds.attrs[key] = attributes[key]

    return ds
