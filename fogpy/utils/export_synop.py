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

""" This module implements export routines for synoptical station data as
shapefile file"""

import fogpy
import os
import osgeo
import osgeo.ogr
import osgeo.osr

from fogpy.utils import import_synop


def create_shpfile(data, outfile, epsg=4326, para=['vis'], nodata=-9999):
    """ Function to export synoptical station data as ESRI shape file"""
    # Init spatial reference locally
    spatialReference = osgeo.osr.SpatialReference()
    # Define this reference to be the EPSG code
    spatialReference.ImportFromEPSG(int(epsg))
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile')
    # Create export file
    shapeData = driver.CreateDataSource(outfile)
    # Create a corresponding layer for our data with given spatial information.
    layer = shapeData.CreateLayer('layer', spatialReference, osgeo.ogr.wkbPoint)
    # Gets parameters of the current shapefile
    layer_defn = layer.GetLayerDefn()
    index = 0
    fielddict = {'name': 0, 'altitude': 1, 'lat': 2, 'lon': 3}
    fields = ['name', 'altitude', 'lat', 'lon'] + para
    addindex = 4
    for ele in para:
        fielddict[ele] = addindex
        addindex += 1
    # Create new fields with the content of read synop data
    for field in fields:
        new_field = osgeo.ogr.FieldDefn(field, osgeo.ogr.OFTString)
        layer.CreateField(new_field)
    # Loop over stations and add them as vector points
    for row in data:
        point = osgeo.ogr.Geometry(osgeo.ogr.wkbPoint)
        point.AddPoint(row[3], row[2])
        feature = osgeo.ogr.Feature(layer_defn)
        feature.SetGeometry(point)  # Set the coordinates
        feature.SetFID(index)
        for field in fields:
            i = feature.GetFieldIndex(field)
            if row[fielddict[field]] is None:
                val = nodata
            else:
                val = row[fielddict[field]]
            try:
                feature.SetField(i, val)
            except:
                Warning("Index: {} - Value {} of type: {} can't be added"
                        .format(i, val, type(val)))
                feature.SetField(i, None)
        layer.CreateFeature(feature)
        index += 1
    shapeData.Destroy()  # Close the shapefile


def main():
    shpfile = '/tmp/FLS/stations_20131112080000.shp'
    base = os.path.split(fogpy.__file__)
    synopfile = os.path.join(base[0], '..', 'etc', 'result_20131112.bufr')
    metarfile = os.path.join(base[0], '..', 'etc', 'result_20131112_metar.bufr')
    swisfile = os.path.join(base[0], '..', 'etc', 'result_20131112_swis.bufr')
    synops = import_synop.read_synop(synopfile, 'visibility')
    metars = import_synop.read_metar(metarfile, 'visibility', latlim=(47, 56),
                                     lonlim=(4, 16))
    swis = import_synop.read_swis(swisfile, 'visibility')
    input = synops['20131112080000'] + metars['20131112083000'] + \
        swis['20131112083000']
    create_shpfile(input, shpfile)

if __name__ == '__main__':
    main()
