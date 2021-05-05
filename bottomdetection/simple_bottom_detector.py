# -*- coding: utf-8 -*-
"""
A simple threshold-based bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""
import warnings

import numpy as np
import xarray as xr

from bottomdetection.parameters import Parameters


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    channel_sv = zarr_data['sv'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    depth_ranges, indices = detect_bottom_single_channel(channel_sv, threshold_sv, parameters.minimum_range)

    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    depth_ranges_back_step, indices_back_step = back_step(channel_sv, indices, heave_corrected_transducer_depth,
                                                          0.001, parameters.maximum_backstep_distance)

    bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step - parameters.offset
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
    m = channel_sv.where((~np.isnan(channel_sv)) & (channel_sv >= threshold), other=-np.inf)
    offset: int = max(int((minimum_range - channel_sv.range[0]) / (channel_sv.range[1] - channel_sv.range[0])), 0)
    bottom_indices = m[:, offset:].argmax(axis=1) + offset
    bottom_depths = channel_sv.range[bottom_indices].where(bottom_indices != offset, np.nan)
    bottom_indices = bottom_indices.where(bottom_indices != offset, -1)
    return xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name='bottom_index', data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})


def _shift(arr, num, fill_value=np.nan):
    # faster than ndimage.shift
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def _back_step_inner(v, v_prev, v_next, shift, shift_prev, shift_next, di, vi, min_depth_value_fraction: float, max_offset):
    if di < 0:
        return -1
    # stack previous and next array and take max
    shift_prev = int(shift_prev) if not np.isnan(shift_prev) else shift
    shift_next = int(shift_next) if not np.isnan(shift_next) else shift
    # ignore warnings when all arrays have nan in the same position
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        vs = np.nanmax(np.stack([v, _shift(v_prev, shift_prev - shift), _shift(v_next, shift_next - shift)]), axis=0)
    # stack shifted arrays in range direction and take max
    a = np.nanmax(np.stack([np.asarray(vs[di - max_offset:di + 1])[::-1],
                            np.asarray(_shift(vs, 1)[di - max_offset:di + 1])[::-1],
                            np.asarray(_shift(vs, -1)[di - max_offset:di + 1])[::-1]]), axis=0)
    back_step_offset = (a <= min_depth_value_fraction * vi).argmax()
    back_step_offset = back_step_offset if back_step_offset > 0 else max_offset
    #forward step on an unsmoothed array
    forward_step_offset = 0
    if v[di - back_step_offset] < min_depth_value_fraction * vi:
        forward_step_offset = (np.asarray(v[di - back_step_offset + 1:di]) > min_depth_value_fraction * vi).argmax()
    return di - back_step_offset + forward_step_offset


def back_step(sv_array: xr.DataArray, depths_indices: xr.DataArray, depth_correction, min_depth_value_fraction: float,
              maximum_distance=10):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param depth_correction: the recorded depth correction (heave plus transducer draft)
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :param maximum_distance: a maximal distance above bottom accepted as the minimal bottom distance
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    max_offset: int = int(maximum_distance / (sv_array.range[1] - sv_array.range[0]))
    range_shift = np.round(depth_correction / (sv_array.range[1] - sv_array.range[0])).astype(int)

    back_step_indices = xr.apply_ufunc(_back_step_inner,
                                       sv_array,
                                       sv_array.shift(ping_time=1),
                                       sv_array.shift(ping_time=-1),
                                       range_shift,
                                       range_shift.shift(ping_time=1),
                                       range_shift.shift(ping_time=-1),
                                       depths_indices,
                                       sv_array[:, depths_indices],
                                       input_core_dims=[['range'], ['range'], ['range'],
                                                        [], [], [], [], []],
                                       kwargs={'min_depth_value_fraction': min_depth_value_fraction,
                                               'max_offset': max_offset},
                                       vectorize=True,
                                       dask='parallelized',
                                       output_dtypes=[np.int64]
                                       )

    bottom_depths = sv_array.range[back_step_indices].where(back_step_indices >= 0, np.nan)
    return xr.DataArray(name='bottom_depth_backstep', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name='bottom_index_backstep', data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
