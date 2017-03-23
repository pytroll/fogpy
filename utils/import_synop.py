#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016

# Author(s):
#   Thomas Leppelt <thomas.leppelt@dwd.de>

# This file is part of the fogpy package.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" This module implements import routines for synoptical station data as 
bufr file"""

import logging
from trollbufr.bufr import Bufr
from trollbufr import load_file
from datetime import datetime

trollbufr_logger = logging.getLogger('trollbufr')
trollbufr_logger.setLevel(logging.CRITICAL)


def read_synop(file, param, min=None, max=None):
    """ Reading bufr files for synoptical station data and provide dictionary
    with weather data for cloud base height and visibility.
    The results are subsequently filtered by cloud base height and visibility

    Arguments:
        file    Bufr file with synop reports
        param    Name of the parameter that will be extracted
        min    Threshold for minimum value of parameter
        max    Threshold for maximum value of parameter

    Returns list of station dictionaries for given thresholds
    """
    result = {}
    bfr = Bufr("libdwd", "/data/tleppelt/git/trollbufr/")
    for blob, size, header in load_file.next_bufr(file):
        bfr.decode(blob)
        try:
            for subset in bfr.next_subset():
                gotit = 0
                stationdict = {}
                for k, m, (v, q) in subset.next_data():
                    if k == 1015:  # Station name
                        stationdict['name'] = v.strip()
                    if k == 5001:  # Latitude
                        stationdict['lat'] = v
                    if k == 6001:  # Longitude
                        stationdict['lon'] = v
                    if k == 7030:  # Altitude
                        stationdict['altitude'] = v
                    elif k == 4001:  # Year
                        stationdict['year'] = v
                    elif k == 4002:  # Month
                        stationdict['month'] = v
                    elif k == 4003:  # Day
                        stationdict['day'] = v
                    elif k == 4004:  # Hour
                        stationdict['hour'] = v
                    elif k == 20003:  # Present weather
                        stationdict['present weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20013:  # Cloud base height
                        if v is not None:
                            if 'cbh' in stationdict.keys():
                                stationdict['cbh'] += v
                            else:
                                stationdict['cbh'] = v
                    elif k == 2001:  # Auto/manual measurement
                        # 1 - 3 : Manual human observations. Manned stations
                        # 0, 4 - 7 : Only automatic observations
                        stationdict['type'] = v
                    elif k == 20001:  # Visibility
                        stationdict['visibility'] = v
                    elif k == 1002:  # WMO station number
                        stationdict['wmo'] = v
                # Apply thresholds
                stationtime = datetime(stationdict['year'],
                                       stationdict['month'],
                                       stationdict['day'],
                                       stationdict['hour']
                                       ).strftime("%Y%m%d%H%M%S")

                if min is not None and stationdict[param] < min:
                    continue
                elif max is not None and stationdict[param] >= max:
                    continue
                elif stationdict[param] is None:
                    continue

                if stationtime in result.keys():
                    result[stationtime].append([stationdict['name'],
                                                stationdict['lat'],
                                                stationdict['lon'],
                                                stationdict[param]])
                else:
                    result[stationtime] = [[stationdict['name'],
                                           stationdict['lat'],
                                           stationdict['lon'],
                                           stationdict[param]]]
        except Exception, e:
            print("ERROR: Unresolved station request: {}".format(e))
    return(result)

#fogdict = read_synop(bufr_file, 'visibility', max=1000)
#for k in fogdict.keys():
#    print(k, fogdict[k])
