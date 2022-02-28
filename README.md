# CRIMAC-classifiers-bottom

This repository contains code for bottom detection in the CRIMAC project.

The bottom detection works on zarr files created by
[CRIMAC-preprocessing](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing).

## Running using Docker

Mounted directories:
1. `/in_dir` - Input directory containing zarr data.
2. `/out_dir` - Output directory where the annotation file will be written.
3. `/work_dir` - Directory with LSSS work files.
   Needed only when `ALGORITHM` is set to `work_files`.

Options as environment variables:
1. `INPUT_NAME` - Name of the zarr file in `in_dir`.
2. `OUTPUT_NAME` - Name of the annotation file in `out_dir`.
   The output format is given by the file name suffix.
   * Pandas DataFrame:
      * `.csv`
      * `.html`
      * `.parquet`
   * Xarray Dataset:
      * `.nc`
      * `.zarr`
3. `ALGORITHM` - Optional. The bottom detection algorithm to use:
   * `angles` - For data with sloping bottom. Used by the `combined` algorithm.
   * `combined` - A combination of the `edge` and `angles` algorithms.
   * `constant` - A very fast algorithm for testing and debugging.
   * `edge` - For data with flat bottom. Used by the `combined` algorithm.
   * `simple` - Backstepping from maximum s<sub>v</sub>.
   * `work_files` - Uses the lowest layer boundary in LSSS work files as the detected bottom.
     The `/work_dir` directory must be mounted.

   See more details in a separate document on the [bottom detection algorithms](doc/BottomDetectionAlgorithms.md).
4. Algorithm parameters. Optional.
   * `PARAMETER_minimum_range` \[m] - The minimum range of the detected bottom.
   * `PARAMETER_offset` \[m] - Additional offset to the bottom after backstepping.
   * `PARAMETER_threshold_log_sv` \[dB] - The minimum S<sub>v</sub> value for detecting bottom.

Example:
```bash
docker run -it --name bottomdetection \
  -v /home/user/data:/in_dir \
  -v /home/user/output:/out_dir  \
  --env INPUT_NAME=dataset.zarr \
  --env OUTPUT_NAME=bottom.parquet \
  --env ALGORITHM=simple \
  --env PARAMETER_offset=0.5 \
  crimac/bottomdetection
```
