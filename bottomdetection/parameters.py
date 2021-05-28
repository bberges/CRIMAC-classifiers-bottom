# -*- coding: utf-8 -*-
"""
Parameters for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""


class Parameters:
    minimum_range = 10
    offset = 0.5
    threshold_log_sv = -31

    def __str__(self):
        return '\n'.join([f'{attr} = {getattr(self, attr)}' for attr in dir(self) if not attr.startswith('__')])
