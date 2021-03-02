# -*- coding: utf-8 -*-
"""
Data structures for the annotation masks in CRIMAC-annotationtools.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import pandas as pd


class Annotation:
    def __init__(self):
        self.pingTime = []
        self.mask_depth_upper = []
        self.mask_depth_lower = []
        self.priority = []
        self.acousticCat = []
        self.proportion = []
        self.ID = []
        self.ChannelID = []

    def __str__(self):
        return f'pingTime: {len(np.unique(self.pingTime))}' \
               f', acousticCat: {np.unique(self.acousticCat)}' \
               f', proportion: {np.unique(self.proportion)}' \
               f', ID: {len(np.unique(self.ID))}'

    def append(self,
               pingTime,
               mask_depth_upper,
               mask_depth_lower,
               priority,
               acousticCat,
               proportion,
               ID,
               ChannelID):
        self.pingTime.append(pingTime)
        self.mask_depth_upper.append(mask_depth_upper)
        self.mask_depth_lower.append(mask_depth_lower)
        self.priority.append(priority)
        self.acousticCat.append(acousticCat)
        self.proportion.append(proportion)
        self.ID.append(ID)
        self.ChannelID.append(ChannelID)

    def to_data_frame(self):
        df = pd.DataFrame(data={'pingTime': self.pingTime,
                                'mask_depth_upper': self.mask_depth_upper,
                                'mask_depth_lower': self.mask_depth_lower,
                                'priority': self.priority,
                                'acousticCat': self.acousticCat,
                                'proportion': self.proportion,
                                'ID': self.ID,
                                'ChannelID': self.ChannelID})

        return df.astype({'acousticCat': 'int64', 'proportion': 'float64', 'ID': 'string', 'ChannelID': 'string'})
