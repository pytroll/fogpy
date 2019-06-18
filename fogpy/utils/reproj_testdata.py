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

""" This module provide utilities to reproject the test data """

import os
import numpy as np
import fogpy
from datetime import datetime
from pyresample import image, geometry
from pyresample import utils
from satpy.scene import Scene
from satpy.dataset import Dataset
from trollimage.colormap import Colormap
from satpy.utils import debug_on

debug_on()
# Define geos projection boundaries for Germany.
# area_extent: (x_ll, y_ll, x_ur, y_ur)
germ_areadef = utils.load_area('/home/mastho/git/satpy/satpy/etc/areas.def',
                               'germ')
euro_areadef = utils.load_area('/home/mastho/git/satpy/satpy/etc/areas.def',
                               'euro4')
germ_extent = germ_areadef.area_extent_ll
print(dir(germ_extent))
geos_areadef = utils.load_area('/home/mastho/git/satpy/satpy/etc/areas.def',
                               'EuropeCanary')
print(germ_extent[0::2], germ_extent[1::2])
print(list(germ_extent[1::2]))
x, y = geos_areadef.get_xy_from_lonlat(list(germ_extent[0::2]),
                                       list(germ_extent[1::2]))
print(x, y)
print(y[0])
xproj, yproj = geos_areadef.get_proj_coords()
xll, xur = xproj[y, x]
yll, yur = yproj[y, x]
new_extent = (xll, yll, xur, yur)
print(new_extent)
area_id = 'Ger_geos'
name = 'Ger_geos'
proj_id = 'Ger_geos'
proj4_args = ("proj=geos, lat_0=0.0, lon_0=0, a=6378144.0, "
              "b=6356759.0, h=35785831.0, rf=295.49")
x_size = 298
y_size = 141
proj_dict = {'a': '6378144.0', 'b': '6356759.0', 'units': 'm', 'lon_0': '0',
             'h': '35785831.0', 'lat_0': '0', 'rf': '295.49',
             'proj': 'geos'}
area_def = geometry.AreaDefinition(area_id, name, proj_id, proj_dict, x_size,
                                   y_size, new_extent)

print(area_def)

# Import test data
base = os.path.split(fogpy.__file__)
testfile = os.path.join(base[0], '..', 'etc', 'fog_testdata.npy')
testdata = np.load(testfile)

# Load test data
inputs = np.dsplit(testdata, 13)
ir108 = inputs[0]
ir039 = inputs[1]
vis008 = inputs[2]
nir016 = inputs[3]
vis006 = inputs[4]
ir087 = inputs[5]
ir120 = inputs[6]
elev = inputs[7]
cot = inputs[8]
reff = inputs[9]
cwp = inputs[10]
lat = inputs[11]
lon = inputs[12]

msg_con_quick = image.ImageContainerQuick(ir108.squeeze(), area_def)
area_con_quick = msg_con_quick.resample(euro_areadef)
result_data_quick = area_con_quick.image_data

# Create satpy scene
testscene = Scene(platform_name="msg",
                  sensor="seviri",
                  start_time=datetime(2013, 11, 12, 8, 30),
                  end_time=datetime(2013, 11, 12, 8, 45),
                  area=area_def)
array_kwargs = {'area': area_def}

testscene['ir108'] = Dataset(ir108.squeeze(), **array_kwargs)
print(testscene['ir108'])
testscene.show(
        'ir108',
        overlay={'coast_dir': '/home/mastho/data/', 'color': 'gray'})
resampscene = testscene.resample('germ')
print(resampscene.shape)

# Define custom fog colormap
fogcol = Colormap((0., (250 / 255.0, 200 / 255.0, 40 / 255.0)),
                  (1., (1.0, 1.0, 229 / 255.0)))
maskcol = (250 / 255.0, 200 / 255.0, 40 / 255.0)
