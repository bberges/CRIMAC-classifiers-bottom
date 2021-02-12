# -*- coding: utf-8 -*-
"""
Data structures for the annotation masks in CRIMAC-annotationtools.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

region_type_enum = {u'empty_water': 0, u'no_data': 1, u'analysis': 2, u'track': 3, u'marker': 4}


class Annotation:
    def __init__(self, info, mask):
        """
        :param info:
        :param mask: list of AnnotationMask
        """
        self.info = info
        self.mask = mask

    def __str__(self):
        return "info: " + str(self.info) \
               + "\nmasks: " \
               + "\n       ".join(str(m) for m in self.mask)


class AnnotationInfo:
    def __init__(self, channel_names, ping_time):
        """
        :param channel_names: list containing the name of each channel
        :param ping_time: must be np.array
        """
        self.channel_names = channel_names
        self.ping_time = ping_time
        self.numberOfPings = len(ping_time)
        self.timeFirstPing = ping_time[0]

    def __str__(self):
        return "channels: " + str(self.channel_names) \
               + ", number_of_pings: " + str(self.numberOfPings)


class AnnotationMask:
    def __init__(self,
                 region_id,
                 region_name,
                 region_provenance,
                 region_type,
                 region_channels,
                 region_category_ids,
                 region_category_proportions,
                 start_time,
                 end_time,
                 min_depth,
                 max_depth,
                 mask_times,
                 priority,
                 mask_depth):
        """

        :param region_id:
        :param region_name:
        :param region_provenance:
        :param region_type: region_type_enum
        :param region_channels:
        :param region_category_ids:
        :param region_category_proportions:
        :param start_time:
        :param end_time:
        :param min_depth:
        :param max_depth:
        :param mask_times:
        :param priority:
        :param mask_depth:
        """
        self.region_id = region_id
        self.region_name = region_name
        self.region_provenance = region_provenance
        self.region_type = region_type
        self.region_channels = region_channels
        self.region_category_ids = region_category_ids
        self.region_category_proportions = region_category_proportions
        self.start_time = start_time
        self.end_time = end_time
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mask_times = mask_times
        self.priority = priority
        self.mask_depth = mask_depth

    def __str__(self):
        return "id: " + self.region_id \
               + ", name: " + self.region_name \
               + ", number_of_pings: " + str(len(self.mask_times))
