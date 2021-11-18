import numpy as np
import xarray as xr

import warnings

SOUND_VELOCITY = 1500  # not available in zarr-file?

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


def first_bottom_index(sv, threshold, depth_correction, pulse_duration, min_index):
    m = sv.where((~np.isnan(sv)) & (sv >= threshold), other=-np.inf)
    sample_dist = float(sv.range[1] - sv.range[0])
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


def _bottom_width_inner(sv, sv_prev, sv_next, shift, shift_prev, shift_next, bottom_index, factor):
    if bottom_index < 0:
        return np.nan
    max_sv_value = sv[bottom_index]
    sv_array = stack_max(sv, sv_prev, sv_next, shift, shift_prev, shift_next)

    ia = np.argmax(sv_array[:bottom_index][::-1] <= max_sv_value * factor)
    ib = np.argmax(sv_array[bottom_index:] <= max_sv_value * factor)
    return np.float32(ia + ib + 1)


def bottom_width(sv_array: xr.DataArray, depths_indices: xr.DataArray, depth_correction, factor):

    range_shift = np.round(depth_correction / (sv_array.range[1] - sv_array.range[0])).astype(int)

    data = xr.apply_ufunc(_bottom_width_inner,
                          sv_array,
                          sv_array.shift(ping_time=1),
                          sv_array.shift(ping_time=-1),
                          range_shift,
                          range_shift.shift(ping_time=1),
                          range_shift.shift(ping_time=-1),
                          depths_indices,
                          input_core_dims=[['range'], []],
                          kwargs={'factor': factor},
                          vectorize=True,
                          dask='parallelized',
                          output_dtypes=[np.float64]
                          )
    return xr.DataArray(name='bottom_width', data=data, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})


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
    bottom_indices = first_bottom_index(channel_sv, threshold, depth_correction, pulse_duration, offset)

    bottom_depths = channel_sv.range[bottom_indices].where(bottom_indices > 0, np.nan)
    return xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time}), \
           xr.DataArray(name='bottom_index', data=bottom_indices, dims=['ping_time'],
                        coords={'ping_time': channel_sv.ping_time})