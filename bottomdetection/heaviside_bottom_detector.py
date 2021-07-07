
import numpy as np
import xarray as xr

from bottomdetection.parameters import Parameters
from bottomdetection.simple_bottom_detector import detect_bottom_single_channel
from bottomdetection.simple_bottom_detector import _shift


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    channel_sv = zarr_data['sv'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    depth_ranges, indices = detect_bottom_single_channel(channel_sv, threshold_sv, parameters.minimum_range)

    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    depth_ranges_back_step, indices_back_step = back_step_hs(channel_sv, indices,
                                                             0.001, 3e-3, minimum_range=parameters.minimum_range)

    bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step - parameters.offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': zarr_data['ping_time']})
    bottom_depths = bottom_depths.dropna('ping_time')
    return bottom_depths


def _back_step_inner_hs(v, di, vi, heaviside, min_depth_value_fraction, abs_threshold, start_index):
    if di < 0:
        return -1

    hs_half_length = len(heaviside) // 2

    segment_end = di

    segment_start = start_index
    array_start = max(0, segment_start - hs_half_length)
    array_end = min(segment_end + hs_half_length, len(v))
    start_offset = array_start - segment_start + hs_half_length
    # stack shifted arrays in range direction and take max
    a = np.nanmax(np.stack([np.asarray(v[array_start:array_end]),
                            np.asarray(_shift(v, 1)[array_start:array_end]),
                            np.asarray(_shift(v, -1)[array_start:array_end])]), axis=0)

    # convolve a with heaviside function, and find max
    a_hs = np.convolve(a, np.flip(heaviside), 'valid')

    # find corresponding array aligned with the valid part of a_hs
    center_array = a[hs_half_length:hs_half_length + len(a_hs)]

    threshold = max(vi * min_depth_value_fraction, abs_threshold)

    valid_idx = np.flatnonzero(center_array <= threshold)
    if len(valid_idx) == 0:
        return 0

    # forward step on an un-smoothed array
    unsmoothed = v[array_start + hs_half_length:array_start + hs_half_length + len(a_hs)]
    last_idx = valid_idx[-1]
    if unsmoothed[last_idx] <= threshold:
        d = np.diff(unsmoothed[last_idx:] <= threshold)
        idx, = d.nonzero()
        if len(idx) > 0 and idx[0] > 0:
            addition = np.arange(start=last_idx + 1, stop=last_idx + 1 + idx[0], step=1, dtype=np.int32)
            valid_idx = np.append(valid_idx, addition)

    back_step_index = valid_idx[a_hs[valid_idx].argmax()]
    return back_step_index + start_index + start_offset


def back_step_hs(sv_array: xr.DataArray, depths_indices: xr.DataArray, min_depth_value_fraction: float,
                 abs_backstep_threshold: float, alpha=0.95, minimum_range=10):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :param abs_backstep_threshold: an absolute backstep threshold value
    :param minimum_range: the minimum range from the transducer
    :param alpha: factor of the importance of having low values above compared to having high values below the bottom
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    offset: int = max(int((minimum_range - sv_array.range[0]) / (sv_array.range[1] - sv_array.range[0])), 0)

    hs_half_length = 5
    heaviside = np.repeat(np.array([-alpha, 1.0 - alpha]), hs_half_length)
    heaviside = np.insert(heaviside, hs_half_length, 1.0 - alpha)

    back_step_indices = xr.apply_ufunc(_back_step_inner_hs,
                                       sv_array,
                                       depths_indices,
                                       sv_array[:, depths_indices],
                                       input_core_dims=[['range'],
                                                        [], []],
                                       kwargs={'heaviside': heaviside,
                                               'min_depth_value_fraction': min_depth_value_fraction,
                                               'abs_threshold': abs_backstep_threshold,
                                               'start_index': offset},
                                       vectorize=True,
                                       dask='parallelized',
                                       output_dtypes=[np.int64]
                                       )

    bottom_depths = sv_array.range[back_step_indices].where(back_step_indices >= 0, np.nan)
    return xr.DataArray(name='bottom_depth_backstep', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name='bottom_index_backstep', data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
