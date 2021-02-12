# CRIMAC-classifiers-bottom

This repository contains code for bottom detection in the CRIMAC project.

The bottom detection takes input from zarr files created by
[CRIMAC-preprocessing](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-preprocessing),
and outputs bottom as annotation masks using
[CRIMAC-annotationtools](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-annotationtools).

## Running using Docker

Two directories must be mounted:
1. `/in_dir` - Input directory containing zarr data.
2. `/out_dir` - Output directory where the annotation file will be written.

Options as environment variables:
1. `INPUT_NAME` - Name of the zarr file in `in_dir`.
2. `OUTPUT_NAME` - Name of the annotation file in `out_dir`.
   The output format is given by the file name suffix:
    * `.nc` - netCDF4.
    * `.work` - LSSS work file.
3. `ALGORITHM` - The bottom detection algorithm to use:
    * `constant` - A very fast algorithm for testing and debugging.
    * `simple` - A very simple threshold based algorithm.

Example:
```bash
docker run -it --name bottomdetection \
  -v /home/user/data:/in_dir \
  -v /home/user/output:/out_dir  \
  --env INPUT_NAME=dataset.zarr \
  --env OUTPUT_NAME=bottom.nc \
  --env ALGORITHM=simple \
  crimac/bottomdetection
```
