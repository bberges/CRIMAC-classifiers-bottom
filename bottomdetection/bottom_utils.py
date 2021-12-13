# -*- coding: utf-8 -*-
"""
Common methods for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr
import scipy.signal as si

import warnings

SOUND_VELOCITY = 1500.0  # not yet available in zarr-file
BEAM_WIDTH_ALONGSHIP = {
    '18000': 11.0,
    '38000': 7.1,
    '70000': 7.1,
    '120000': 7.1,
    '200000': 7.1,
    '364000': 7.1
}
BEAM_WIDTH_ATHWARTSHIP = {
    '18000': 11.0,
    '38000': 7.1,
    '70000': 7.1,
    '120000': 7.1,
    '200000': 7.1,
    '364000': 7.1
}


def shift_arr(arr, num, fill_value=np.nan):
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


def stack_max(v, v_prev, v_next, shift, shift_prev, shift_next):
    # stack previous and next array and take max
    shift_prev = int(shift_prev) if not np.isnan(shift_prev) else shift
    shift_next = int(shift_next) if not np.isnan(shift_next) else shift
    # ignore warnings when all arrays have nan in the same position
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        return np.nanmax(np.stack([v, shift_arr(v_prev, shift_prev - shift), shift_arr(v_next, shift_next - shift)]), axis=0)


def find_peaks(data, threshold):
    return si.find_peaks(data, threshold=threshold)


def peak_prominences(data, peaks):
    return si.peak_prominences(data, peaks)


def peak_widths(data, peaks, prominences):
    return si.peak_widths(data, peaks, prominence_data=prominences)


def gauss_derivative(i, center_index, b):
    x = i - center_index
    return x * np.exp(-x * x / b)


def create_gauss_derivative_kernel(n, center_index):
    b = np.square(center_index / 2.0)
    result = np.asarray([gauss_derivative(i, center_index, b) for i in range(n)])
    return result


def _mean_at_index(sv, index, radius):
    if index < 0:
        return np.nan
    return np.nanmean(sv[max(0, index - radius):min(len(sv), index + radius)])


def mean_at_index(sv, indices, radius):
    data = xr.apply_ufunc(_mean_at_index,
                          sv,
                          indices,
                          input_core_dims=[['range'], []],
                          kwargs={'radius': radius},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='mean_at_index', data=data, dims=['ping_time'],
                        coords={'ping_time': sv.ping_time})


def first_bottom_index(sv, sample_dist, threshold, depth_correction, pulse_duration, min_index):
    m = sv.where((~np.isnan(sv)) & (sv >= threshold), other=-np.inf)
    offset: int = min_index
    bottom_indices = m[:, offset:].argmax(axis=1) + offset
    # check if max sv is at second bottom echo
    sound_velocity = SOUND_VELOCITY
    check_indices = np.rint(bottom_indices / 2 - depth_correction / (2 * sample_dist)).where(bottom_indices != offset, -1).astype(np.int64)
    pulse_thickness = int(np.rint(pulse_duration * sound_velocity / (2 * sample_dist)))
    check_sum = mean_at_index(sv, check_indices, pulse_thickness)
    max_sv_sum = mean_at_index(sv, bottom_indices, pulse_thickness)
    differences = check_sum > max_sv_sum * 0.75
    best_indices = check_indices.where(differences, bottom_indices)
    return best_indices.where(best_indices != offset, -1)


def possibly_small_backstep(sv, threshold, index):
    if index > 0 and sv[index] > threshold:
        backstep_index = index
        for i in range(max(0, index - 1), max(0, index - 2), -1):
            if sv[i] < sv[backstep_index] * 0.75:
                backstep_index = i
        index = backstep_index
    return index


def min_bottom_thickness(sv, frequency, start_index, sample_dist, pulse_duration, bottom_index):
    if bottom_index < 0:
        return np.nan

    pulse_thickness = pulse_duration * SOUND_VELOCITY / 2.0

    if start_index < len(sv):
        bottom_range = bottom_index * sample_dist
        alpha = max(BEAM_WIDTH_ALONGSHIP.get(frequency, 7.1), BEAM_WIDTH_ATHWARTSHIP.get(frequency, 7.1)) / 2.0
        return pulse_thickness + bottom_range * (1.0 / np.cos(np.radians(alpha)) - 1)

    return pulse_thickness


def _bottom_width_inner(sv, bottom_index, factor, frequency, start_index, sample_dist, pulse_duration):
    if bottom_index < 0:
        return np.nan

    begin_index = bottom_index // 2
    end_index = min(3 * bottom_index // 2, len(sv))
    cumulative = np.cumsum(sv[begin_index:end_index])
    sv_sum = np.nansum(sv[begin_index:end_index])

    bottom_begin = np.searchsorted(cumulative, sv_sum * factor) + begin_index
    bottom_end = np.searchsorted(cumulative, sv_sum * (1 - factor)) + begin_index

    quantile_thickness = np.float32(bottom_end - bottom_begin + 1)
    minimum_thickness_meters = min_bottom_thickness(sv, frequency, start_index, sample_dist, pulse_duration, bottom_index)
    minimum_thickness = int(np.round(minimum_thickness_meters / sample_dist))
    return max(quantile_thickness, minimum_thickness)


def bottom_width(sv_array: xr.DataArray, depths_indices: xr.DataArray, factor, frequency, pulse_duration, minimum_range=10.0):

    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    start_index: int = max(int((minimum_range - sv_array.range[0]) / sample_dist), 0)

    data = xr.apply_ufunc(_bottom_width_inner,
                          sv_array,
                          depths_indices,
                          input_core_dims=[['range'],
                                           []],
                          kwargs={'factor': factor,
                                  'frequency': frequency,
                                  'start_index': start_index,
                                  'sample_dist': sample_dist,
                                  'pulse_duration': pulse_duration},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='bottom_width', data=data, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})


def _stack_pings_inner(sv, sv_prev, sv_next, shift, shift_prev, shift_next):
    return stack_max(sv, sv_prev, sv_next, shift, shift_prev, shift_next)


def stack_pings(sv_array: xr.DataArray, depth_correction):
    sample_dist = float(sv_array.range[1] - sv_array.range[0])
    range_shift = np.round(depth_correction / sample_dist).astype(np.int32)
    data = xr.apply_ufunc(_stack_pings_inner,
                          sv_array,
                          sv_array.shift(ping_time=1),
                          sv_array.shift(ping_time=-1),
                          range_shift,
                          range_shift.shift(ping_time=1),
                          range_shift.shift(ping_time=-1),
                          input_core_dims=[['range'], ['range'], ['range'],
                                           [], [], []],
                          vectorize=True,
                          dask='parallelized',
                          output_core_dims=[['range']],
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='stacked', data=data, dims=['ping_time', 'range'],
                        coords={'ping_time': sv_array.ping_time, 'range': sv_array.range})


def detect_bottom_single_channel(channel_sv: xr.DataArray, threshold: float, depth_correction, pulse_duration, minimum_range=10.0):
    """
    Detect bottom depths on one channel in the sv-array

    :param channel_sv: an array of log-sv values
    :param threshold: a minimum threshold for bottom depth strength
    :param depth_correction: the recorded depth correction (heave plus transducer draft)
    :param pulse_duration: the pulse duration
    :param minimum_range: the minimum range from the transducer
    :return: a data array of bottom depths and the indices of the bottom depths
    """
    sample_dist = float(channel_sv.range[1] - channel_sv.range[0])
    offset: int = max(int((minimum_range - channel_sv.range[0]) / sample_dist), 0)
    sample_dist = float(channel_sv.range[1] - channel_sv.range[0])

    bottom_indices = first_bottom_index(channel_sv, sample_dist, threshold, depth_correction, pulse_duration, offset)

    bottom_depths = channel_sv.range[bottom_indices].where(bottom_indices > 0, np.nan)
    return xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name='bottom_index', data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})
