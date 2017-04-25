#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017
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

""" This module implements satellite image based fog and low stratus
detection and forecasting algorithm as a PyTROLL custom composite object.
"""

import logging

from algorithms import DayFogLowStratusAlgorithm
from mpop.imageo.geo_image import GeoImage
from trollimage.colormap import Colormap

logger = logging.getLogger(__name__)

# Define custom fog colormap
fogcol = Colormap((1., (0.0, 0.0, 0.0)),
                  (0., (250 / 255.0, 200 / 255.0, 40 / 255.0)))


def fls_day(self, elevation, cot, reff, lwp=None, cth=None):
    """ This method defines a composite for fog and low stratus detection
    and forecasting at daytime. The fog algorithm is optimized for the
    Meteosat Second Generation - SERVIRI instrument.

    Required additional inputs:
        elevation    Ditital elevation model as array
        cot    Cloud optical thickness(depth) as array
        reff    Cloud particle effective radius as array
        lwp    Liquid water path as array
        cth    Cloud top height as array, optional
    """
    logger.debug("Creating fog composite for {} instrument scene {}"
                 .format(self.fullname, self.time_slot))

    self.check_channels(0.635, 0.81, 1.64, 3.92, 8.7, 10.8, 12.0)

    # Get central lon/lat coordinates for the image
    area = self[10.8].area
    lon, lat = area.get_lonlats()

    flsinput = {'vis006': self[0.635].data,
                'vis008': self[0.81].data,
                'ir108': self[10.8].data,
                'nir016': self[1.64].data,
                'ir039': self[3.92].data,
                'ir120': self[12.0].data,
                'ir087': self[8.7].data,
                'lat': lat,
                'lon': lon,
                'time': self.time_slot,
                'elev': elevation,
                'cot': cot,
                'reff': reff,
                'lwp': lwp,
                'cth': cth,
                'plot': True,
                'save': True,
                'dir': '/tmp',
                'resize': '5'}

    # Compute fog mask
    flsalgo = DayFogLowStratusAlgorithm(**flsinput)
    fls, mask = flsalgo.run()

    # Create geoimage object from algorithm result
    flsimg = GeoImage(fls, area, self.time_slot,
                      fill_value=0, mode="L")
    flsimg.enhance(stretch="crude")

    maskimg = GeoImage(~mask, area, self.time_slot,
                       fill_value=0, mode="L")
    maskimg.enhance(stretch="crude")

    return flsimg, maskimg

fls_day.prerequisites = set([0.635, 0.81, 1.64, 3.92, 8.7, 10.8, 12.0])


def fls_night(self):
    """ This method defines a composite for fog and low stratus detection
    and forecasting at night."""
    pass

# List of composites for SEVIRI instrument
seviri = [fls_day, fls_night]
