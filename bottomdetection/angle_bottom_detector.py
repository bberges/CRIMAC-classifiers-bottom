import numpy as np
import xarray as xr

from bottomdetection.parameters import Parameters
from bottomdetection import bottom_utils


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    channel_sv = zarr_data['sv'][channel_index]
    angle_alongship = zarr_data['angle_alongship'][channel_index]
    angle_athwartship = zarr_data['angle_athwartship'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    pulse_duration = float(zarr_data['pulse_length'][channel_index])
    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    stacked_sv = bottom_utils.stack_pings(channel_sv, heave_corrected_transducer_depth)

    depth_ranges, indices = bottom_utils.detect_bottom_single_channel(stacked_sv, threshold_sv, heave_corrected_transducer_depth,
                                                                      pulse_duration, parameters.minimum_range)

    bottom_ranges = bottom_utils.bottom_range(stacked_sv, indices, 0.1)

    angle_alongship_filtered = bottom_utils.filter_angles(channel_sv, angle_alongship, heave_corrected_transducer_depth)
    angle_athwartship_filtered = bottom_utils.filter_angles(channel_sv, angle_athwartship, heave_corrected_transducer_depth)

    depth_ranges_angles, indices_angles, qualities = detect_from_angles(angle_alongship_filtered, angle_athwartship_filtered, bottom_ranges)

    # returning the candidate with the highest quality
    bottom_depths = heave_corrected_transducer_depth + depth_ranges_angles[:, 0] - parameters.offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': channel_sv['ping_time']})
    bottom_depths = bottom_depths.dropna('ping_time')
    return bottom_depths


def _local_top_value(values, i):
    n = 3
    ia = max(i - n, 0)
    ib = min(i + n + 1, len(values))
    va = np.mean(values[ia:i])
    vb = np.mean(values[i+1:ib])
    return 2 * values[i] - (va + vb)  # -second derivative


def _find_bottom_index(i_begin, i_end, r_squared):
    r_squared_begin = r_squared[i_begin]
    v = np.asarray([abs(r_squared[i] - r_squared_begin) * _local_top_value(r_squared, i) for i in range(i_begin, i_end - 2)])
    return np.argmax(v) + i_begin if len(v) > 0 else i_begin


def _find_local_minima_before(i, minima_sorted):
    idx_peak_before = np.searchsorted(minima_sorted, i, side='left')
    if idx_peak_before > 0:
        return minima_sorted[idx_peak_before - 1]
    else:
        return 0


def _find_local_minima_after(i, minima_sorted):
    idx_peak_after = np.searchsorted(minima_sorted, i, side='right')
    if idx_peak_after < len(minima_sorted):
        return minima_sorted[idx_peak_after]
    else:
        return np.nan


def _new_bottom_candidate(i_convolution_peak, i_end, convolution, convolution_minima, r_squared):
    i_bottom = _find_bottom_index(i_convolution_peak, i_end, r_squared)

    idx_peak_before = _find_local_minima_before(i_convolution_peak, convolution_minima)
    idx_peak_after = _find_local_minima_after(i_convolution_peak, convolution_minima)
    idx_peak_after = len(convolution) - 1 if np.isnan(idx_peak_after) else idx_peak_after
    delta_before = convolution[i_convolution_peak] - convolution[idx_peak_before]
    delta_after = convolution[i_convolution_peak] - convolution[idx_peak_after]
    delta_quality = bottom_utils.clamp((delta_before + delta_after) / 2, 0, 1)

    r_squared_quality = r_squared[i_bottom]
    total_quality = delta_quality * r_squared_quality

    return i_bottom, total_quality


def _make_candidates(indices, max_index, i_end, convolution, r_squared):
    convolution_minima = bottom_utils.local_minima(convolution, 1)
    candidate_indices = []
    candidate_qualities = []
    idx, q = _new_bottom_candidate(max_index, i_end if len(indices) == 0 else indices[0], convolution, convolution_minima, r_squared)
    candidate_indices.append(idx)
    candidate_qualities.append(q)
    for i, elem in enumerate(indices):
        next_i = i_end if len(indices) <= i + 1 else indices[i+1]
        idx, q = _new_bottom_candidate(elem, next_i, convolution, convolution_minima, r_squared)
        candidate_indices.append(idx)
        candidate_qualities.append(q)
    return candidate_indices, candidate_qualities


def _detect_from_angles_inner(angles_alongship, angles_athwartship, bottom_range, sample_dist, max_candidates):
    begin_index = bottom_range[0]
    end_index = bottom_range[1]

    lin_regress = LinearRegression()
    along_r_squared = np.flip(np.asarray([lin_regress.add_point(i, angles_alongship[i]) for i in np.linspace(end_index, 0, end_index + 1, dtype=np.int32)]))
    lin_regress = LinearRegression()
    athwart_r_squared = np.flip(np.asarray([lin_regress.add_point(i, angles_athwartship[i]) for i in np.linspace(end_index, 0, end_index + 1, dtype=np.int32)]))
    along_r_squared = np.where(np.isnan(along_r_squared), 0, along_r_squared)
    athwart_r_squared = np.where(np.isnan(athwart_r_squared), 0, athwart_r_squared)

    i_test_min = max(0, begin_index - (end_index - begin_index))

    along_fits_best = np.max(along_r_squared[i_test_min:begin_index]) > np.max(athwart_r_squared[i_test_min:begin_index])
    best_r_squared = along_r_squared if along_fits_best else athwart_r_squared

    k = int(np.round(2.0 / sample_dist))
    kernel = bottom_utils.step_kernel(k)

    convolution = np.convolve(best_r_squared, np.flip(kernel), 'same')
    convolution[begin_index:] = 0
    arg_max_conv = np.argmax(convolution)
    max_conv = convolution[arg_max_conv]
    convolution /= max_conv

    peaks = bottom_utils.local_maxima(convolution, 1)
    peaks = peaks[(peaks > arg_max_conv) & (peaks < begin_index)].tolist()
    candidate_indices, qualities = _make_candidates(peaks, arg_max_conv, begin_index, convolution, best_r_squared)
    candidate_indices, candidate_qualities = bottom_utils.sorted_candidates(max_candidates, candidate_indices, qualities, 0)

    return np.concatenate([candidate_indices, candidate_qualities])


def detect_from_angles(angles_alongship, angles_athwartship, ranges):
    """
    Find minimum bottom depths looking at split beam angles
    :param angles_alongship: array of split beam angles in the alongship direction
    :param angles_athwartship: array of split beam angles in the athwartship direction
    :param ranges: the width of the bottom echo in number of samples
    :return: data arrays of bottom depths, the indices of the bottom depths, and quality of bottom depths
    """
    sample_dist = float(angles_alongship.range[1] - angles_alongship.range[0])

    max_candidates = 3
    indices = xr.apply_ufunc(_detect_from_angles_inner,
                             angles_alongship,
                             angles_athwartship,
                             ranges,
                             input_core_dims=[['range'],
                                              ['range'],
                                              ['begin_end']],
                             kwargs={'sample_dist': sample_dist,
                                     'max_candidates': max_candidates
                                     },
                             vectorize=True,
                             dask='parallelized',
                             output_core_dims=[['candidates']],
                             dask_gufunc_kwargs={'output_sizes': {'candidates': max_candidates*2}},
                             output_dtypes=[np.float32]
                             )

    bottom_depths = angles_alongship.range[indices[:, :max_candidates].astype(np.int64)].where(indices[:, :max_candidates] >= 0, np.nan)

    return xr.DataArray(name='bottom_depth_angles', data=bottom_depths, dims=['ping_time', 'candidates'],
                        coords={'ping_time': angles_alongship.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='bottom_index_angles', data=indices[:, :max_candidates].astype(np.int64), dims=['ping_time', 'candidates'],
                        coords={'ping_time': angles_alongship.ping_time, 'candidates': range(max_candidates)}), \
           xr.DataArray(name='quality', data=indices[:, max_candidates:max_candidates*2], dims=['ping_time', 'candidates'],
                        coords={'ping_time': angles_alongship.ping_time, 'candidates': range(max_candidates)})


class LinearRegression:

    def __init__(self):
        self.x_bar = 0
        self.y_bar = 0
        self.sum_x = 0
        self.sum_y = 0
        self.sum_xx = 0
        self.sum_yy = 0
        self.sum_xy = 0
        self.n = 0

    def add_point(self, x, y):
        if self.n == 0:
            self.x_bar = x
            self.y_bar = y
        else:
            f1 = 1.0 + self.n
            f2 = self.n / (1.0 + self.n)
            dx = x - self.x_bar
            dy = y - self.y_bar
            self.sum_xx += dx * dx * f2
            self.sum_yy += dy * dy * f2
            self.sum_xy += dx * dy * f2
            self.x_bar += dx / f1
            self.y_bar += dy / f1
        self.sum_x += x
        self.sum_y += y
        self.n += 1
        return self.r_square()

    def total_sum_squares(self):
        if self.n < 2:
            return np.nan
        return self.sum_yy

    def sum_squared_errors(self):
        if self.n < 2:
            return np.nan
        return max(0.0, self.sum_yy - self.sum_xy * self.sum_xy / self.sum_xx)

    def r_square(self):
        ssto = self.total_sum_squares()
        if np.isnan(ssto) or ssto == 0:
            return np.nan
        return (ssto - self.sum_squared_errors()) / ssto
