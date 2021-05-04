# -*- coding: utf-8 -*-
"""
The entry point when running in a Docker container.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import os
import sys
import dask

from dask.distributed import Client

from bottomdetection import bottom_detection_main

if __name__ == "__main__":
    if os.getenv('DEBUG', 'false') == 'true':
        print('Press enter...')
        input()
        sys.exit(0)

    input_name = os.getenv('INPUT_NAME', '.')
    output_name = os.getenv('OUTPUT_NAME', 'out.parquet')
    algorithm = os.getenv('ALGORITHM', 'simple')

    in_dir = os.path.expanduser("/in_dir")
    out_dir = os.path.expanduser("/out_dir")

    dask.config.set({'temporary_directory': out_dir})
    client = Client()
    print(client)

    bottom_detection_main.run(zarr_file=in_dir + '/' + input_name,
                              out_file=out_dir + '/' + output_name,
                              bottom_algorithm=algorithm)
