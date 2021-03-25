# -*- coding: utf-8 -*-
"""
The main function for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import xarray as xr
import pandas as pd

from bottomdetection import bottom_annotation
from bottomdetection import simple_bottom_detector


def run(zarr_file: str, out_file: str, bottom_algorithm: str):
    zarr_data = xr.open_zarr(zarr_file, chunks={'frequency': 'auto', 'ping_time': 'auto', 'range': -1})
    print(zarr_data)

    bottom_depth = detect_bottom(zarr_data, bottom_algorithm)
    print('\n\nBottom depth:')
    print(bottom_depth)

    annotation = bottom_annotation.to_bottom_annotation(zarr_data['channelID'][0].values, bottom_depth)
    print('\n\nAnnotation:')
    print(annotation)

    write_to_file(annotation, out_file)


def detect_bottom(zarr_data: xr.Dataset, bottom_algorithm: str):
    if bottom_algorithm == 'simple':
        return simple_bottom_detector.detect_bottom(zarr_data)

    if bottom_algorithm.startswith('constant'):
        # A very fast algorithm for testing and debugging.
        depth = float(bottom_algorithm[8:]) if len(bottom_algorithm) > 8 else 100
        bottom_depth = np.full(len(zarr_data['ping_time']), depth)
        return xr.DataArray(name='bottom_depth', data=bottom_depth, dims=['ping_time'],
                            coords={'ping_time': zarr_data['ping_time']})

    raise ValueError('Unknown bottom algorithm: ' + bottom_algorithm)


def write_to_file(df: pd.DataFrame, file: str):
    if file.endswith('.csv'):
        df.to_csv(file)

    elif file.endswith('.html'):
        df.to_html(file)

    elif file.endswith('.parquet'):
        table = pa.Table.from_pandas(df)
        with pq.ParquetWriter(file, table.schema) as writer:
            writer.write_table(table=table)

    else:
        raise ValueError('Unknown out format: ' + file)
