# -*- coding: utf-8 -*-
"""
A simple threshold-based bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr


def detect_bottom(zarr_data):
    sv = zarr_data.sv
    log_sv = 10 * np.log10(sv)
    depths, indices = detect_bottom_single_channel(log_sv[0], -31)
    depths_back_step, indices_back_step = back_step(sv[0], indices, 0.001)
    return depths_back_step.data


def detect_bottom_single_channel(channel_sv: xr.DataArray, threshold: float, minimum_range=10):
    """
    Detect bottom depths on one channel in the sv-array

    :param channel_sv: an array of log-sv values
    :param threshold: a minimum threshold for bottom depth strength
    :param minimum_range: the minimum range from the transducer
    :return: a data array of bottom depths and the indices of the bottom depths
    """
    m = np.ma.masked_where(((np.isnan(channel_sv)) | (channel_sv < threshold)), channel_sv)
    offset: int = int(minimum_range / (channel_sv.range[1] - channel_sv.range[0]))
    bottom_indices = m[:, offset:].argmax(axis=1) + offset
    bottom_depths = np.where(bottom_indices == offset, np.nan, channel_sv.range[bottom_indices])
    bottom_indices = np.where(bottom_indices == offset, -1, bottom_indices)
    return xr.DataArray(name="bottom_depth", data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name="bottom_index", data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})


def back_step(sv_array: xr.DataArray, depths_indices: xr.DataArray, min_depth_value_fraction: float, maximum_range=10):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :param maximum_range: a maximal distance above bottom accepted as the minimal bottom distance
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    values = sv_array[:, depths_indices]
    max_offset: int = int(maximum_range / (sv_array.range[1] - sv_array.range[0]))
    back_step_indices = []
    for i, v in enumerate(sv_array):
        back_step_index = -1
        if depths_indices[i].values > 0:
            back_step_offset = (np.asarray(v[depths_indices[i].values - max_offset:depths_indices[
                i].values + 1])[::-1] < min_depth_value_fraction * values[i].values).argmax()
            back_step_index = depths_indices[i].values - back_step_offset \
                if back_step_offset > 0 else int(depths_indices[i].values - max_offset)
        back_step_indices.append(int(back_step_index))
    back_step_indices = np.asarray(back_step_indices)
    bottom_depths = np.where(back_step_indices < 0, np.nan, sv_array.range[back_step_indices])
    return xr.DataArray(name="bottom_depth_backstep", data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name="bottom_index_backstep", data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
