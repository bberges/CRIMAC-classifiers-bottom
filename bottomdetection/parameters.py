# -*- coding: utf-8 -*-
"""
Parameters for bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

from dataclasses import dataclass


@dataclass
class Parameters:
    minimum_range: float = 10.0
    offset: float = 0.5
    threshold_log_sv: float = -31.0
