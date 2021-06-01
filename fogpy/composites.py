#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Fogpy developers

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

"""Interface Fogpy functionality as Satpy composite.

This module implements satellite image based fog and low stratus
detection and forecasting algorithm as a Satpy custom composite object.
"""

import logging
import numpy
import xarray
import pathlib

import appdirs
import satpy
import satpy.composites
import satpy.dataset
import pyorbital.astronomy
import pkg_resources

from .algorithms import DayFogLowStratusAlgorithm
from .algorithms import NightFogLowStratusAlgorithm
from .utils import dl_dem


logger = logging.getLogger(__name__)


class FogCompositor(satpy.composites.GenericCompositor):
    """A compositor for fog.

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
    def _convert_xr_to_ma(projectables):
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

        fv = numpy.nan
        # convert to xarray images
        dims = projectables[0].dims
        coords = projectables[0].coords
        attrs = {k: projectables[0].attrs[k]
                 for k in {"satellite_longitude", "satellite_latitude",
                           "satellite_altitude", "sensor", "platform_name",
                           "orbital_parameters", "georef_offset_corrected",
                           "start_time", "end_time", "area", "resolution"} &
                 projectables[0].attrs.keys()}

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
        dem = pathlib.Path(appdirs.user_data_dir("fogpy")) / path_dem
        if not dem.exists():
            dl_dem(dem)
        filenames = [dem]
        self.elevation = satpy.Scene(reader="generic_image",
                               filenames=filenames)
        self.elevation.load(["image"])
        return super().__init__(*args, **kwargs)

    def _verify_requirements(self, optional_datasets):
        """Verify that required cloud microphysics present

        Can be either cmic_cot/cmic_lwp/cmic_reff or cot/lwp/reff.
        """
        D = {}
        needs = {"cot": {"cot", "cmic_cot"},
                 "lwp": {"lwp", "cwp", "cmic_lwp"},
                 "reff": {"reff", "cmic_lwp"}}
        for x in optional_datasets:
            for (n, p) in needs.items():
                if x.attrs["name"] in p:
                    D[n] = x
                    continue
        missing = needs.keys() - D.keys()
        if missing:
            raise ValueError("Missing fog inputs: " + ", ".join(missing))
        return D

    def __call__(self, projectables, *args, optional_datasets, **kwargs):
        D = self._verify_requirements(optional_datasets)
        (area, lat, lon) = self._get_area_lat_lon(projectables)

        # fogpy is still working with masked arrays and does not yet support
        # xarray / dask (see #6).  For now, convert to masked arrays.
        maskproj = self._convert_xr_to_ma(projectables)
        D = dict(zip(D.keys(), self._convert_xr_to_ma(D.values())))

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
                    'cot': D["cot"],
                    'reff': D["reff"],
                    'lwp': D["lwp"],
                    "cwp": D["lwp"]}
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
            ds.attrs[k] = self.attrs.get(k)

        return ds


class FogCompositorDay(satpy.composites.GenericCompositor):
    def __call__(self, projectables, *args, **kwargs):
        # in the yaml file, fls_day has as a single prerequisite
        # _intermediate_fls_day.  Therefore, the first and only
        # projectable is actually a Dataset, and pass a DataArray
        # to the superclass.__call__ method.
        ds = projectables[0]
        # the fogpy algorithm has the mask as True where fog is absent and
        # False where fog is present, although that is OK for a masked array,
        # we want the opposite interpretation when visualising it as the A band
        # on a LA-type image, therefore invert the truthiness with a unary

        # normally we'd invert this with ~x, but due ta a bug this loses the
        # attributes: see https://github.com/pydata/xarray/issues/4065
        # True^x is equivalent to ~x for booleans
        with xarray.set_options(keep_attrs=True):
            return super().__call__((ds["fls_day"], True ^ ds["fls_mask"]), *args, **kwargs)


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

        maskproj = self._convert_xr_to_ma(projectables)

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


def save_extras(sc, fn):
    """Save the `fls_days_extra` dataset to NetCDF

    The ``fls_day_extra`` dataset as produced by the `FogCompositorDayExtra` and
    loaded using ``.load(["fls_day_extra"])`` is unique in the sense that it is
    an `xarray.Dataset` rather than an `xarray.DataArray`.  This means it can't
    be stored with the usual satpy routines.  Because some of its attributes
    contain special types, it can`t be stored with `Dataset.to_netcdf` either.

    This function transfers the data variables as direct members of a new
    `Scene` object and then use the `cf_writer` to write those to a NetCDF file.

    Args:
        sc : Scene
            Scene object with the already loaded ``fls_day_extra`` "composite"
        fn : str-like or path
            Path to which to write NetCDF
    """
    s = satpy.Scene()
    ds = sc["fls_day_extra"]
    for k in ds.data_vars:
        s[k] = ds[k]
    s.save_datasets(
            writer="cf",
            datasets=ds.data_vars.keys(),
            filename=str(fn))
