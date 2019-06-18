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

import fogpy
import logging
import os
import os.path

from fogpy.lowwatercloud import CloudLayer as CL
from trollbufr.bufr import Bufr
from trollbufr import load_file
from datetime import datetime

trollbufr_logger = logging.getLogger('trollbufr')
trollbufr_logger.setLevel(logging.CRITICAL)


def read_synop(file, params, min=None, max=None):
    """ Reading bufr files for synoptical station data and provide dictionary
    with weather data for cloud base height and visibility.
    The results are subsequently filtered by cloud base height and visibility

    Arguments:
        file    Bufr file with synop reports
        params    List of parameter names that will be extracted
        min    Threshold for minimum value of parameter
        max    Threshold for maximum value of parameter

    Returns list of station dictionaries for given thresholds
    """
    result = {}
    bfr = Bufr("libdwd", os.getenv("BUFR_TABLES"))
    for blob, size, header in load_file.next_bufr(file):
        bfr.decode(blob)
        try:
            for subset in bfr.next_subset():
                gotit = 0
                stationdict = {}
                for (k, m, v, q) in subset.next_data():
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
                    elif k == 4005:  # Hour
                        stationdict['minute'] = v
                    elif k == 20003:  # Present weather
                        stationdict['present weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20004:  # Past weather
                        stationdict['past weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20013:  # Cloud base height
                        if v is not None:
                            if 'cbh' in stationdict.keys():
                                if stationdict['cbh'] > v:
                                    stationdict['cbh'] = v
                            else:
                                stationdict['cbh'] = v
                        else:
                            stationdict['cbh'] = None
                    elif k == 2001:  # Auto/manual measurement
                        # 1 - 3 : Manual human observations. Manned stations
                        # 0, 4 - 7 : Only automatic observations
                        stationdict['type'] = v
                    elif k == 20001:  # Visibility
                        stationdict['visibility'] = v
                    elif k == 12101:  # Mean air temperature in K
                        stationdict['air temperature'] = v
                    elif k == 12103:  # Dew point temperature in K
                        stationdict['dew point'] = v
                    elif k == 20010:  # Cloud cover in %
                        stationdict['cloudcover'] = v
                    elif k == 13003:  # Relative humidity in %
                        stationdict['relative humidity'] = v
                    elif k == 11001:  # Wind direction in degree
                        stationdict['wind direction'] = v
                    elif k == 11002:  # Wind speed in m s-1
                        stationdict['wind speed'] = v
                    elif k == 1002:  # WMO station number
                        stationdict['wmo'] = v
                # Apply thresholds
                stationtime = datetime(stationdict['year'],
                                       stationdict['month'],
                                       stationdict['day'],
                                       stationdict['hour'],
                                       stationdict['minute'],
                                       ).strftime("%Y%m%d%H%M%S")
                paralist = []
                if not isinstance(params, list):
                    params = [params]
                for param in params:
                    if param not in stationdict:
                        res = None
                    elif min is not None and stationdict[param] < min:
                        res = None
                    elif max is not None and stationdict[param] >= max:
                        res = None
                    elif stationdict[param] is None:
                        res = None
                    else:
                        res = stationdict[param]
                    paralist.append(res)
                if all([i is None for i in paralist]):
                    continue
                # Add station data to result list
                if stationtime in result.keys():
                    result[stationtime].append([stationdict['name'],
                                                stationdict['altitude'],
                                                stationdict['lat'],
                                                stationdict['lon']] + paralist)
                else:
                    result[stationtime] = [[stationdict['name'],
                                           stationdict['altitude'],
                                           stationdict['lat'],
                                           stationdict['lon']] + paralist]
        except Exception as e:
            "ERROR: Unresolved station request: {}".format(e)
    return(result)


def read_metar(file, params, min=None, max=None, latlim=None, lonlim=None):
    """ Reading bufr files for METAR station data and provide dictionary
    with weather data for cloud base height and visibility.
    The results are subsequently filtered by cloud base height and visibility

    Arguments:
        file    Bufr file with synop reports
        params    List of parameter names that will be extracted
        min    Threshold for minimum value of parameter
        max    Threshold for maximum value of parameter
        latlim Tuple of minimum and maximum latitude values for valid result
        lonlim Tuple of minimum and maximum longitude values for valid result

    Returns list of station dictionaries for given thresholds
    """
    result = {}
    bfr = Bufr("libdwd", os.getenv("BUFR_TABLES"))
    for blob, size, header in load_file.next_bufr(file):
        bfr.decode(blob)
        try:
            for subset in bfr.next_subset():
                gotit = 0
                stationdict = {}
                for (k, m, v, q) in subset.next_data():
                    if k == 1063:  # Station name
                        stationdict['name'] = v.strip()
                    if k == 5002:  # Latitude
                        stationdict['lat'] = v
                    if k == 6002:  # Longitude
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
                    elif k == 4005:  # Hour
                        stationdict['minute'] = v
                    elif k == 20003:  # Present weather
                        stationdict['present weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20004:  # Past weather
                        stationdict['past weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20013:  # Cloud base height
                        if v is not None:
                            if 'cbh' in stationdict.keys():
                                if stationdict['cbh'] > v:
                                    stationdict['cbh'] = v
                            else:
                                stationdict['cbh'] = v
                        else:
                            stationdict['cbh'] = None
                    elif k == 2001:  # Auto/manual measurement
                        # 1 - 3 : Manual human observations. Manned stations
                        # 0, 4 - 7 : Only automatic observations
                        stationdict['type'] = v
                    elif k == 20060:  # Prevailing visibility
                        stationdict['visibility'] = v
                    elif k == 12023:  # Mean air temperature in °C
                        stationdict['air temperature'] = CL.check_temp(v, 'kelvin')
                    elif k == 12024:  # Dew point temperature in °C
                        stationdict['dew point'] = CL.check_temp(v, 'kelvin')
                    elif k == 20010:  # Cloud cover in %
                        stationdict['cloudcover'] = v
                    elif k == 13003:  # Relative humidity in %
                        stationdict['relative humidity'] = v
                    elif k == 11001:  # Wind direction in degree
                        stationdict['wind direction'] = v
                    elif k == 11002:  # Wind speed in m s-1
                        stationdict['wind speed'] = v
                    elif k == 1002:  # WMO station number
                        stationdict['wmo'] = v
                    elif k == 1024:  # WMO station number
                        stationdict['coords'] = v
                # Apply thresholds
                stationtime = datetime(stationdict['year'],
                                       stationdict['month'],
                                       stationdict['day'],
                                       stationdict['hour'],
                                       stationdict['minute'],
                                       ).strftime("%Y%m%d%H%M%S")
                paralist = []
                if not isinstance(params, list):
                    params = [params]
                for param in params:
                    if param not in stationdict:
                        res = None
                    elif min is not None and stationdict[param] < min:
                        res = None
                    elif max is not None and stationdict[param] >= max:
                        res = None
                    elif stationdict[param] is None:
                        res = None
                    else:
                        res = stationdict[param]
                    paralist.append(res)
                if all([i is None for i in paralist]):
                    continue
                # Test for limited coordinates
                if latlim is not None:
                    if stationdict['lat'] < latlim[0]:
                        continue
                    elif stationdict['lat'] > latlim[1]:
                        continue
                if lonlim is not None:
                    if stationdict['lon'] < lonlim[0]:
                        continue
                    elif stationdict['lon'] > lonlim[1]:
                        continue
                # Add station data to result list
                if stationtime in result.keys():
                    result[stationtime].append([stationdict['name'],
                                                stationdict['altitude'],
                                                stationdict['lat'],
                                                stationdict['lon']] + paralist)
                else:
                    result[stationtime] = [[stationdict['name'],
                                           stationdict['altitude'],
                                           stationdict['lat'],
                                           stationdict['lon']] + paralist]
        except Exception as e:
            "ERROR: Unresolved station request: {}".format(e)
    return(result)


def read_swis(file, params, min=None, max=None, latlim=None, lonlim=None):
    """ Reading bufr files for street weather information system data and
    provide dictionary with weather data for cloud base height and visibility.
    The results are subsequently filtered by cloud base height and visibility

    Arguments:
        file    Bufr file with synop reports
        params    List of parameter names that will be extracted
        min    Threshold for minimum value of parameter
        max    Threshold for maximum value of parameter
        latlim Tuple of minimum and maximum latitude values for valid result
        lonlim Tuple of minimum and maximum longitude values for valid result

    Returns list of station dictionaries for given thresholds
    """
    result = {}
    bfr = Bufr("libdwd", os.getenv("BUFR_TABLES"))
    for blob, size, header in load_file.next_bufr(file):
        bfr.decode(blob)
        try:
            for subset in bfr.next_subset():
                gotit = 0
                stationdict = {}
                for (k, m, v, q) in subset.next_data():
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
                    elif k == 4005:  # Hour
                        stationdict['minute'] = v
                    elif k == 20003:  # Present weather
                        stationdict['present weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                        # 28 refers to fog or ice fog from manned station
                        # Reference: FM 94, table: 4677
                    elif k == 20004:  # Past weather
                        stationdict['past weather'] = v
                        # Values from 40 to 49 are refering to fog and ice fog
                        # Patchy fog or fog edges value 11 or 12
                    elif k == 20013:  # Cloud base height
                        if v is not None:
                            if 'cbh' in stationdict.keys():
                                if stationdict['cbh'] > v:
                                    stationdict['cbh'] = v
                            else:
                                stationdict['cbh'] = v
                        else:
                            stationdict['cbh'] = None
                    elif k == 2001:  # Auto/manual measurement
                        # 1 - 3 : Manual human observations. Manned stations
                        # 0, 4 - 7 : Only automatic observations
                        stationdict['type'] = v
                    elif k == 20001:  # Prevailing visibility
                        if v is not None:
                            stationdict['visibility'] = v * 10
                        else:
                            stationdict['visibility'] = v
                    elif k == 12101:  # Mean air temperature in K
                        stationdict['air temperature'] = v
                    elif k == 12103:  # Dew point temperature in K
                        stationdict['dew point'] = v
                    elif k == 20010:  # Cloud cover in %
                        stationdict['cloudcover'] = v
                    elif k == 13003:  # Relative humidity in %
                        stationdict['relative humidity'] = v
                    elif k == 11001:  # Wind direction in degree
                        stationdict['wind direction'] = v
                    elif k == 11002:  # Wind speed in m s-1
                        stationdict['wind speed'] = v
                    elif k == 1002:  # WMO station number
                        stationdict['wmo'] = v
                    elif k == 1024:  # WMO station number
                        stationdict['coords'] = v
                    elif k == 33005:  # WMO station number
                        stationdict['quality'] = v
                # Apply thresholds
                stationtime = datetime(stationdict['year'],
                                       stationdict['month'],
                                       stationdict['day'],
                                       stationdict['hour'],
                                       stationdict['minute'],
                                       ).strftime("%Y%m%d%H%M%S")
                paralist = []
                if not isinstance(params, list):
                    params = [params]
                for param in params:
                    if param not in stationdict:
                        res = None
                    elif min is not None and stationdict[param] < min:
                        res = None
                    elif max is not None and stationdict[param] >= max:
                        res = None
                    elif stationdict[param] is None:
                        res = None
                    else:
                        res = stationdict[param]
                    paralist.append(res)
                if all([i is None for i in paralist]):
                    continue
                # Test for limited coordinates
                if latlim is not None:
                    if stationdict['lat'] < latlim[0]:
                        continue
                    elif stationdict['lat'] > latlim[1]:
                        continue
                if lonlim is not None:
                    if stationdict['lon'] < lonlim[0]:
                        continue
                    elif stationdict['lon'] > lonlim[1]:
                        continue
                # Add station data to result list
                if stationtime in result.keys():
                    result[stationtime].append([stationdict['name'],
                                                stationdict['altitude'],
                                                stationdict['lat'],
                                                stationdict['lon']] + paralist)
                else:
                    result[stationtime] = [[stationdict['name'],
                                           stationdict['altitude'],
                                           stationdict['lat'],
                                           stationdict['lon']] + paralist]
        except Exception as e:
            "ERROR: Unresolved station request: {}".format(e)
    return(result)


def main():
    base = os.path.split(fogpy.__file__)
    synopfile = os.path.join(base[0], '..', 'etc', 'result_20131112.bufr')
    metarfile = os.path.join(base[0], '..', 'etc', 'result_20131112_metar.bufr')
    swisfile = os.path.join(base[0], '..', 'etc', 'result_20131112_swis.bufr')
    print(read_synop(synopfile, ['visibility']))
    print(read_metar(metarfile, 'visibility', latlim=(45, 60), lonlim=(3, 18)))
    print(read_swis(swisfile, ['visibility']))

if __name__ == '__main__':
    main()
