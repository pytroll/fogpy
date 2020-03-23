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
import xarray

import satpy.composites
import satpy.dataset
import pyorbital.astronomy

from satpy import Scene
from .algorithms import DayFogLowStratusAlgorithm
from .algorithms import NightFogLowStratusAlgorithm


logger = logging.getLogger(__name__)


class FogCompositor(satpy.composites.GenericCompositor):
    """A compositor for fog

    FIXME DOC
    """

    def __init__(self, name,
                 prerequisites=None,
                 optional_prerequisites=None,
                 **kwargs):
        return super().__init__(
                name,
                prerequisites=prerequisites,
                optional_prerequisites=optional_prerequisites,
                **kwargs)

    def _get_area_lat_lon(self, projectables):
        projectables = self.check_areas(projectables)

        # Get central lon/lat coordinates for the image
        area = projectables[0].area
        lon, lat = area.get_lonlats()

        return (area, lat, lon)

    @staticmethod
    def _convert_projectables(projectables):
        """Convert projectables to masked arrays

        fogpy is still working with masked arrays and does not yet support
        xarray / dask (see #6).  For now, convert to masked arrays.  This
        function takes a list (or other iterable) of
        ``:class:xarray.DataArray`` instances and converts this to a list
        of masked arrays.  The mask corresponds to any non-finite data in
        each input data array.

        Args:
            projectables (iterable): Iterable with xarray.DataArray
                instances, such as `:func:satpy.Scene._generate_composite`
                passes on to the ``__call__`` method of each Compositor
                class.

        Returns:
            List of masked arrays, of the same length as ``projectables``,
            each projectable converted to a masked array.
        """

        return [numpy.ma.masked_invalid(p.values, copy=False)
                for p in projectables]

    @staticmethod
    def _convert_ma_to_xr(projectables, *args):
        """Convert fogpy algorithm result to xarray images

        The fogpy algorithms return numpy masked arrays, but satpy
        compositors expect xarray DataArry objects.  This method
        takes the output of the fogpy algorithm routine and converts
        it to an xarray DataArray, with the attributes corresponding
        to a Satpy composite.

        Args:
            projectables (iterable): Iterable with xarray.DataArray
                instances, such as `:func:satpy.Scene._generate_composite`
                passes on to the ``__call__`` method of each Compositor
                class.
            fls (masked_array): Masked array such as returned by
                ``fogpy.algorithms.BaseSatelliteAlgorithm.run`` or its
                subclasses
            mask (masked_array): Mask corresponding to fls.

        Returns:
            List[xarray.DataArray] list of xarray DataArrays, corresponding
            to the ``*args`` inputs passed.  If an image and a mask, those
            can be passed to ``GenericCompositor.__call__`` to get a LA image
            ``xarray.DataArray``, or the latter can be constructed directly.
        """

        fv = 9000
        # convert to xarray images
        dims = projectables[0].dims
        coords = projectables[0].coords
        attrs = {k: projectables[0].attrs[k]
                 for k in ("satellite_longitude", "satellite_latitude",
                           "satellite_altitude", "sensor", "platform_name",
                           "orbital_parameters", "georef_offset_corrected",
                           "start_time", "end_time", "area", "resolution")}

        das = [xarray.DataArray(
                   ma.data if isinstance(ma, numpy.ma.MaskedArray) else ma,
                   dims=dims, coords=coords, attrs=attrs)
               for ma in args]
        for (ma, da) in zip(args, das):
            try:
                da.values[ma.mask] = fv
            except AttributeError:  # no mask
                pass
            da.encoding["_FillValue"] = fv

        return das


class _IntermediateFogCompositorDay(FogCompositor):
    def __init__(self, path_dem, *args, **kwargs):
        self.elevation = Scene(reader="generic_image",
                               filenames=[path_dem])
        self.elevation.load(["image"])
        return super().__init__(*args, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        (area, lat, lon) = self._get_area_lat_lon(projectables)

        # fogpy is still working with masked arrays and does not yet support
        # xarray / dask (see #6).  For now, convert to masked arrays.
        maskproj = self._convert_projectables(projectables)

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
                    'elev': numpy.ma.masked_invalid(
                        elev["image"].sel(bands="L").values, copy=False),
                    'cot': maskproj[7],
                    'reff': maskproj[9],
                    'lwp': maskproj[8],
                    "cwp": maskproj[8]}
        # Compute fog mask
        flsalgo = DayFogLowStratusAlgorithm(**flsinput)
        fls, mask = flsalgo.run()

        (xrfls, xrmsk, xrvmask, xrcbh, xrfbh, xrlcth) = self._convert_ma_to_xr(
                projectables, fls, mask, flsalgo.vcloudmask, flsalgo.cbh,
                flsalgo.fbh, flsalgo.lcth)

        ds = xarray.Dataset({
            "fls_day": xrfls,
            "fls_mask": xrmsk,
            "vmask": xrvmask,
            "cbh": xrcbh,
            "fbh": xrfbh,
            "lcthimg": xrlcth})

        ds.attrs.update(satpy.dataset.combine_metadata(
                xrfls.attrs, xrmsk.attrs, xrvmask.attrs,
                xrcbh.attrs, xrfbh.attrs, xrlcth.attrs))

        # NB: isn't this done somewhere more generically?
        for k in ("standard_name", "name", "resolution"):
            ds.attrs[k] = self.attrs[k]

        return ds


class FogCompositorDay(satpy.composites.GenericCompositor):
    def __call__(self, projectables, *args, **kwargs):
        # in the yaml file, fls_day has as a single prerequisite
        # _intermediate_fls_day.  Therefore, the first and only
        # projectable is actually a Dataset, and pass a DataArray
        # to the superclass.__call__ method.
        ds = projectables[0]
        return super().__call__((ds["fls_day"], ds["fls_mask"]), *args, **kwargs)


class FogCompositorDayExtra(satpy.composites.GenericCompositor):
    def __call__(self, projectables, *args, **kwargs):
        ds = projectables[0]
        ds.attrs["standard_name"] = ds.attrs["name"] = "fls_day_extra"
        return ds


class FogCompositorNight(FogCompositor):

    def __call__(self, projectables, *args, **kwargs):
        (area, lat, lon) = self._get_area_lat_lon(projectables)

        sza = pyorbital.astronomy.sun_zenith_angle(
                projectables[0].start_time, lon, lat)

        maskproj = self._convert_projectables(projectables)

        flsinput = {'ir108': maskproj[1],
                    'ir039': maskproj[0],
                    'sza': sza,
                    'lat': lat,
                    'lon': lon,
                    'time': projectables[0].start_time
                    }

        # Compute fog mask
        flsalgo = NightFogLowStratusAlgorithm(**flsinput)
        fls, mask = flsalgo.run()

        (xrfls, xrmsk) = self._convert_ma_to_xr(projectables, fls, mask)

        return super().__call__((xrfls, xrmsk), *args, **kwargs)
