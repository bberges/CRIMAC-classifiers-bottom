# -*- coding: utf-8 -*-
"""
A simple threshold-based bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr


def detect_bottom(zarr_data: xr.Dataset) -> xr.DataArray:
    sv = zarr_data.sv
    sv0 = sv[0]
    #log_sv = 10 * np.log10(sv)
    threshold_log_sv = -31
    threshold_sv = 10 ** (threshold_log_sv / 10)

    depth_ranges, indices = detect_bottom_single_channel(sv0, threshold_sv)

    depth_ranges_back_step, indices_back_step = back_step(sv0, indices, 0.001)

    offset = 0.5
    bottom_depths = depth_ranges_back_step + zarr_data['heave'] + zarr_data['transducer_draft'][0] - offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': zarr_data['ping_time']})
    bottom_depths = bottom_depths.dropna('ping_time')
    return bottom_depths


def detect_bottom_single_channel(channel_sv: xr.DataArray, threshold: float, minimum_range=10):
    """
    Detect bottom depths on one channel in the sv-array

    :param channel_sv: an array of log-sv values
    :param threshold: a minimum threshold for bottom depth strength
    :param minimum_range: the minimum range from the transducer
    :return: a data array of bottom depths and the indices of the bottom depths
    """
    m = channel_sv.where((~np.isnan(channel_sv)) & (channel_sv >= threshold), other=-1)
    offset: int = max(int((minimum_range - channel_sv.range[0]) / (channel_sv.range[1] - channel_sv.range[0])), 0)
    bottom_indices = m[:, offset:].argmax(axis=1) + offset
    bottom_depths = channel_sv.range[bottom_indices].where(bottom_indices != offset, np.nan)
    bottom_indices = bottom_indices.where(bottom_indices != offset, -1)
    return xr.DataArray(name="bottom_depth", data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name="bottom_index", data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})


def _back_step_inner(v, di, min_depth_value_fraction: float, max_offset):
    vi = v[di]

    if di > 0:
        back_step_offset = (np.asarray(v[di - max_offset:di + 1])[::-1] < min_depth_value_fraction * vi).argmax()
        back_step_index = di - back_step_offset \
            if back_step_offset > 0 else int(di - max_offset)
    else:
        back_step_index = -1
    return back_step_index

def back_step(sv_array: xr.DataArray, depths_indices: xr.DataArray, min_depth_value_fraction: float, maximum_distance=10):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :param maximum_distance: a maximal distance above bottom accepted as the minimal bottom distance
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    max_offset: int = int(maximum_distance / (sv_array.range[1] - sv_array.range[0]))

    back_step_indices = xr.apply_ufunc(_back_step_inner,
                        sv_array,
                        depths_indices,
                        input_core_dims=[["range"], []],
                        kwargs={'min_depth_value_fraction': min_depth_value_fraction, 'max_offset': max_offset},
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=[np.int64]
                        )

    bottom_depths = sv_array.range[back_step_indices].where(back_step_indices >= 0, np.nan)
    return xr.DataArray(name="bottom_depth_backstep", data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name="bottom_index_backstep", data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
