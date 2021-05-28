# CRIMAC-classifiers-bottom

This repository contains code for bottom detection in the CRIMAC project.

The bottom detection works on zarr files created by
[CRIMAC-preprocessing](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing).

## Running using Docker

Two directories must be mounted:
1. `/in_dir` - Input directory containing zarr data.
2. `/out_dir` - Output directory where the annotation file will be written.

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
    * `constant` - A very fast algorithm for testing and debugging.
    * `simple` - A very simple threshold based algorithm.
4. Algorithm parameters. Optional.
   * `PARAMETER_minimum_range` \[m] - The minimum range of the detected bottom.
   * `PARAMETER_offset` \[m] - Additional offset to the bottom after backstepping.
   * `PARAMETER_threshold_log_sv` \[dB] - The minimum Sv value for detecting bottom.

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
