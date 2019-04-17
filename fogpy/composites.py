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
import numpy

from .algorithms import DayFogLowStratusAlgorithm
from .algorithms import NightFogLowStratusAlgorithm
from trollimage.xrimage import XRImage
from trollimage.colormap import Colormap

from satpy import Scene

logger = logging.getLogger(__name__)

# Define custom fog colormap
fogcol = Colormap((1., (0.0, 0.0, 0.0)),
                  (0., (250 / 255.0, 200 / 255.0, 40 / 255.0)))


def fls_day(self, elevation, cot, reff, lwp=None, cth=None, validate=False,
            plot=False, plotdir='/tmp', single=False):
    """This method defines a composite for fog and low stratus detection
    and forecasting at daytime.

    The fog algorithm is optimized for the Meteosat Second Generation
    - SERVIRI instrument.

    Args:
        | elevation (:obj:`ndarray`): Ditital elevation model as array.
        | cot (:obj:`ndarray`): Cloud optical thickness(depth) as array.
        | reff (:obj:`ndarray`): Cloud particle effective radius as array.
        | lwp (:obj:`ndarray`): Liquid water path as array.
        | cth (:obj:`ndarray`): Cloud top height as array, optional.
        | validate (:obj:`bool`): Additional cloud mask output, optional.
        | plot (:obj:`bool`): Save filter and algorithm results as png images.
        | plotdir (:obj:`str`): Path to plotting directory as string.
        | single (:obj:`bool`): Compute lowcloud model single pixelwise.
                                Default is False.

    Returns:
        Infrared image with colorized fog areas and the calculated fog mask.
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
                'plot': plot,
                'save': plot,
                'dir': plotdir,
                'single': single,
                'resize': '1'}

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

    if validate:
        # Get cloud mask image
        vmaskimg = GeoImage(flsalgo.vcloudmask, area, self.time_slot,
                            fill_value=0, mode="L")
        vmaskimg.enhance(stretch="crude")

        # Get cloud base height image
        cbhimg = GeoImage(flsalgo.cbh, area, self.time_slot,
                          fill_value=9999, mode="L")

        # Get fog base height image
        fbhimg = GeoImage(flsalgo.fbh, area, self.time_slot,
                          fill_value=9999, mode="L")

        # Get low cloud top height image
        lcthimg = GeoImage(flsalgo.lcth, area, self.time_slot,
                           fill_value=9999, mode="L")

        return [flsimg, maskimg, vmaskimg, cbhimg, fbhimg, lcthimg]
    else:
        return flsimg, maskimg


def fls_night(self, sza):
    """This method defines a composite for fog and low stratus detection
    and forecasting at night.

    The fog algorithm is optimized for the Meteosat Second Generation
    - SERVIRI instrument.

    Args:
        | sza (:obj:`ndarray`): Satellite zenith angle as array.

    Returns:
        Infrared image with colorized fog areas and the calculated fog mask
    """
    logger.debug("Creating fog composite for {} instrument scene {}"
                 .format(self.fullname, self.time_slot))

    self.check_channels(3.92, 10.8)

    # Get central lon/lat coordinates for the image
    area = self[10.8].area
    lon, lat = area.get_lonlats()

    flsinput = {'ir108': self[10.8].data,
                'ir039': self[3.92].data,
                'sza': sza,
                'lat': lat,
                'lon': lon,
                'time': self.time_slot,
                'plot': True,
                'save': True,
                'dir': '/tmp',
                'resize': '5'}

    # Compute fog mask
    flsalgo = NightFogLowStratusAlgorithm(**flsinput)
    fls, mask = flsalgo.run()

    # Create geoimage object from algorithm result
    flsimg = GeoImage(fls, area, self.time_slot,
                      fill_value=0, mode="L")
    flsimg.enhance(stretch="crude")

    maskimg = GeoImage(~mask, area, self.time_slot,
                       fill_value=0, mode="L")
    maskimg.enhance(stretch="crude")

    return flsimg, maskimg


# Set prerequisites
fls_day.prerequisites = set([0.635, 0.81, 1.64, 3.92, 8.7, 10.8, 12.0])
fls_night.prerequisites = set([3.92, 10.8])

# List of composites for SEVIRI instrument
seviri = [fls_day, fls_night]



#####

import satpy.composites

class FogCompositor(satpy.composites.GenericCompositor):
    """A compositor for fog

    FIXME DOC
    """

    def __init__(self, name,
            prerequisites=None,
            optional_prerequisites=None,
            **kwargs):
        super().__init__(name,
                prerequisites=prerequisites,
                optional_prerequisites=optional_prerequisites,
                **kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        super().__call__(datasets,
                optional_datasets=optional_datasets,
                **info)

class FogCompositorDay(FogCompositor):
    def __init__(self, path_dem, *args, **kwargs):
        self.elevation = Scene(
                reader="generic_image",
                filenames=[path_dem])
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        projectables = self.check_areas(projectables)

        # Get central lon/lat coordinates for the image
        area = projectables[0].area
        lon, lat = area.get_lonlats()

        # fogpy is still working with masked arrays and does not yet support
        # xarray / dask (see #6).  For now, convert to masked arrays.
        maskproj = [
                numpy.ma.masked_invalid(p.values, copy=False)
                for p in projectables]
        elev = self.elevation.resample(area)
        flsinput = {'vis006': maskproj[0],
                    'vis008': maskproj[1],
                    'ir108': maskproj[5],
                    'nir016': maskproj[2],
                    'ir039': maskproj[3],
                    'ir120': maskproj[6],
                    'ir087': maskproj[4],
                    'lat': lat,
                    'lon': lon,
                    'time': projectables[0].start_time,
                    'elev': elev,
                    'cot': maskproj[7],
                    'reff': maskproj[9],
                    'lwp': maskproj[8],
                    "cwp": maskproj[8],
                    #'cth': cth,
                    #'plot': plot,
                    #'save': plot,
                    #'dir': plotdir,
                    #'single': single,
                    #'resize': '1',
                    }

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

        if validate:
            # Get cloud mask image
            vmaskimg = GeoImage(flsalgo.vcloudmask, area, self.time_slot,
                                fill_value=0, mode="L")
            vmaskimg.enhance(stretch="crude")

            # Get cloud base height image
            cbhimg = GeoImage(flsalgo.cbh, area, self.time_slot,
                              fill_value=9999, mode="L")

            # Get fog base height image
            fbhimg = GeoImage(flsalgo.fbh, area, self.time_slot,
                              fill_value=9999, mode="L")

            # Get low cloud top height image
            lcthimg = GeoImage(flsalgo.lcth, area, self.time_slot,
                               fill_value=9999, mode="L")

            return [flsimg, maskimg, vmaskimg, cbhimg, fbhimg, lcthimg]
        else:
            return flsimg, maskimg


class FogCompositorNight(FogCompositor):
    pass
