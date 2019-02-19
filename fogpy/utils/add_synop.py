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

"""This script adds synope data for visibility to given geolocated image"""

import fogpy
import numpy as np
import os
from .import_synop import read_synop
from datetime import datetime
from trollimage.image import Image
from trollimage.colormap import Colormap


# Define custom fog colormap
fogcol = Colormap((0., (250 / 255.0, 200 / 255.0, 40 / 255.0)),
                  (1., (1.0, 1.0, 229 / 255.0)))

maskcol = Colormap((1., (250 / 255.0, 200 / 255.0, 40 / 255.0)))

viscol = Colormap((0., (1.0, 0.0, 0.0)),
                  (5000, (0.7, 0.7, 0.7)))
# Red - Violet - Blue - Green
vis_colset = Colormap((0, (228 / 255.0, 26 / 255.0, 28 / 255.0)),
                      (1000, (152 / 255.0, 78 / 255.0, 163 / 255.0)),
                      (5000, (55 / 255.0, 126 / 255.0, 184 / 255.0)),
                      (10000, (77 / 255.0, 175 / 255.0, 74 / 255.0)))


def add_to_array(arr, area, time, bufr, savedir='/tmp', name=None, mode='L',
                 resize=None, ptsize=None, save=False):
    """Add synoptical reports from stations to provided geolocated image array
    """
    # Create array image
    arrshp = arr.shape[:2]
    print(np.nanmin(arr), np.nanmax(arr))
    arr_img = Image(arr, mode=mode)
    # arr_img = Image(channels=[arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]],
    #                mode='RGB')
    arr_img.stretch('crude')
    arr_img.invert()
    arr_img.colorize(maskcol)
    arr_img.invert()

    # Import bufr
    stations = read_synop(bufr, 'visibility')
    currentstations = stations[time.strftime("%Y%m%d%H0000")]
    lats = [i[2] for i in currentstations]
    lons = [i[3] for i in currentstations]
    vis = [i[4] for i in currentstations]

    # Create array for synop parameter
    visarr = np.empty(arrshp)
    visarr.fill(np.nan)

    x, y = (area.get_xy_from_lonlat(lons, lats))
    vis_ma = np.ma.array(vis, mask=x.mask)
    if ptsize:
        xpt = np.array([])
        ypt = np.array([])
        for i, j in zip(x, y):
            xmesh, ymesh = np.meshgrid(np.linspace(i - ptsize, i + ptsize,
                                                   ptsize * 2 + 1),
                                       np.linspace(j - ptsize, j + ptsize,
                                                   ptsize * 2 + 1))
            xpt = np.append(xpt, xmesh.ravel())
            ypt = np.append(ypt, ymesh.ravel())
        vispt = np.ma.array([np.full(((ptsize * 2 + 1,
                                       ptsize * 2 + 1)), p) for p in vis_ma])
        visarr[ypt.astype(int), xpt.astype(int)] = vispt.ravel()
    else:
        visarr[y.compressed(), x.compressed()] = vis_ma.compressed()
    visarr_ma = np.ma.masked_invalid(visarr)
    station_img = Image(visarr_ma, mode='L')
    station_img.colorize(vis_colset)
    station_img.merge(arr_img)
    if resize is not None:
        station_img.resize((arrshp[0] * resize, arrshp[1] * resize))
    if name is None:
        timestr = time.strftime("%Y%m%d%H%M")
        name = "fog_filter_example_stations_{}.png".format(timestr)
    if save:
        savepath = os.path.join(savedir, name)
        station_img.save(savepath)

    return(station_img)


def add_to_image(image, area, time, bufr, savedir='/tmp', name=None,
                 bgimg=None, resize=None, ptsize=None, save=False):
    """Add synoptical visibility reports from station data to provided
    geolocated image array
    """
    arrshp = image.shape[:2]
    # Add optional background image
    if bgimg is not None:
        # Get background image
        bg_img = Image(bgimg.squeeze(), mode='L', fill_value=None)
        bg_img.stretch("crude")
        bg_img.convert("RGB")
#         bg_img.invert()
        image.merge(bg_img)
    # Import bufr
    stations = read_synop(bufr, 'visibility')
    currentstations = stations[time.strftime("%Y%m%d%H0000")]
    lats = [i[2] for i in currentstations]
    lons = [i[3] for i in currentstations]
    vis = [i[4] for i in currentstations]

    # Create array for synop parameter
    visarr = np.empty(arrshp)
    visarr.fill(np.nan)
    # Define custom fog colormap
    viscol = Colormap((0., (1.0, 0.0, 0.0)),
                      (5000, (0.7, 0.7, 0.7)))
    # Red - Violet - Blue - Green
    vis_colset = Colormap((0, (228 / 255.0, 26 / 255.0, 28 / 255.0)),
                          (1000, (152 / 255.0, 78 / 255.0, 163 / 255.0)),
                          (5000, (55 / 255.0, 126 / 255.0, 184 / 255.0)),
                          (10000, (77 / 255.0, 175 / 255.0, 74 / 255.0)))
    x, y = (area.get_xy_from_lonlat(lons, lats))
    vis_ma = np.ma.array(vis, mask=x.mask)
    if ptsize:
        xpt = np.array([])
        ypt = np.array([])
        for i, j in zip(x, y):
            xmesh, ymesh = np.meshgrid(np.linspace(i - ptsize, i + ptsize,
                                                   ptsize * 2 + 1),
                                       np.linspace(j - ptsize, j + ptsize,
                                                   ptsize * 2 + 1))
            xpt = np.append(xpt, xmesh.ravel())
            ypt = np.append(ypt, ymesh.ravel())
        vispt = np.ma.array([np.full(((ptsize * 2 + 1,
                                       ptsize * 2 + 1)), p) for p in vis_ma])
        visarr[ypt.astype(int), xpt.astype(int)] = vispt.ravel()
    else:
        visarr[y.compressed(), x.compressed()] = vis_ma.compressed()
    visarr_ma = np.ma.masked_invalid(visarr)
    station_img = Image(visarr_ma, mode='L')
    station_img.colorize(vis_colset)
    image.convert("RGB")
    station_img.merge(image)
    if resize is not None:
        station_img.resize((arrshp[0] * resize, arrshp[1] * resize))
    if name is None:
        timestr = time.strftime("%Y%m%d%H%M")
        name = "fog_filter_example_stations_{}.png".format(timestr)
    if save:
        savepath = os.path.join(savedir, name)
        station_img.save(savepath)

    return(station_img)

if __name__ == '__main__':

    from pyresample import geometry
    from scipy import misc

    # Set time stamp
    time = datetime(2013, 11, 12, 8, 30)
    # Import image
    # imgfile = 'LowCloudFilter_201311120830.png'
    imgfile = 'LowCloudFilter_201311120830.png'
    imgdir = '/tmp/FLS'
    resize = 5  # Resize factor of FLS image
    imgpath = os.path.join(imgdir, imgfile)
    arr = misc.imread(imgpath)
    arr = np.ma.masked_where(arr == 0, arr)
    print(arr.shape)
    print(np.min(arr))
    # Get bufr file
    base = os.path.split(fogpy.__file__)
    inbufr = os.path.join(base[0], '..', 'etc', 'result_{}.bufr'
                          .format(time.strftime("%Y%m%d")))
#     bufr_dir = '/data/tleppelt/skydata/'
#     bufr_file = "result_{}".format(time.strftime("%Y%m%d"))
#     inbufr = os.path.join(bufr_dir, bufr_file)

    area_id = "geos_germ"
    name = "geos_germ"
    proj_id = "geos"
    proj_dict = {'a': '6378169.00', 'lon_0': '0.00', 'h': '35785831.00',
                 'b': '6356583.80', 'proj': 'geos', 'lat_0': '0.00'}
    x_size = 298 * resize
    y_size = 141 * resize
    area_extent = (214528.82635591552, 4370087.2110124603, 1108648.9697693815,
                   4793144.0573926577)
    area_def = geometry.AreaDefinition(area_id, name, proj_id, proj_dict,
                                       x_size, y_size, area_extent)
    print(area_def)
    img = Image(arr[:, :, :3], mode='RGB')
    add_to_image(img, area_def, time, inbufr)
