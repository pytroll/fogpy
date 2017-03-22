#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017
# Author(s):
#   Thomas Leppelt <thomas.leppelt@dwd.de>
""" This module test fog detection and forecasting algorithmns implemented in
mpop/satpy"""

import pyproj
import os
import sys
import netCDF4 as nc
import numpy as np
from mpop.satellites import GeostationaryFactory
from mpop.utils import debug_on
from mipp import read_geotiff as gtiff
from pyresample import image
from trollimage.image import Image
from trollimage.colormap import Colormap
from track_cb import get_time_period
from PIL import Image as PILimage
from pyresample import utils
from mpop.imageo.geo_image import GeoImage

#debug_on()


def nc_import(infile, inputdir=os.getcwd()):
    """
    Import infrared sounder profiles (Level 2) in netCDF4 format.

    Keyword arguments:
    inputfile -- NetcDF File name as string.
    inputdir -- Directory containg NetCDF file in string format.

    Returns:
    NetCDF Dataset object
    """
    try:
        ncfile = nc.Dataset(os.path.join(inputdir, infile), 'r',
                            format='NETCDF4')
    except:
        msg = "Could not open NetCDF file <%s>" % (infile)
        ncfile = nc.Dataset(os.path.join(inputdir, infile), 'r',
                            format='NETCDF4')
        if ncfile not in locals():
            msg = "Finally could not open NetCDF file <%s>" % (infile)
            raise ImportError(msg)

    return(ncfile)


def draw_synop(img, area_def, lons, lats, ptsize):
    """ Drawing points to image based on lat lon coordinates utilizing
    pycoast functions"""
    from pycoast.cw_pil import ContourWriter
    cw = ContourWriter()
    draw = cw._get_canvas(img)
    outlinecol = "red"

    try:
        (x, y) = area_def.get_xy_from_lonlat(lons, lats)
    except ValueError, exc:
        print("Point not added (%s)", str(exc))
    else:
        # add_dot
        if ptsize is not None:
            dot_box = [x - ptsize, y - ptsize,
                       x + ptsize, y + ptsize]
            cw._draw_ellipse(draw, dot_box, fill=outlinecol,
                             outline=outlinecol)
            cw._finalize(draw)
    return(img)

# Define custom fog colormap
fogcol = Colormap((0., (0.0, 0.0, 0.8)),
                  (1., (250 / 255.0, 200 / 255.0, 40 / 255.0)))

# Define geos projection boundaries for Germany.
# area_extent: (x_ll, y_ll, x_ur, y_ur)
prj = pyproj.Proj(proj='geos', lon_0=0.0, a=6378144.0, b=6356759.0,
                  h=35785831.0, rf=295.49)
x_ll, y_ll = prj(3, 47)
x_ur, y_ur = prj(19, 55)
ger_extent = (x_ll, y_ll, x_ur, y_ur)
germ_areadef = utils.load_area('/data/tleppelt/Pytroll/config/areas.def',
                               'germ')
germ_extent = germ_areadef.area_extent_ll
x_ll, y_ll = prj(*germ_extent[0:2])
x_ur, y_ur = prj(*germ_extent[2:])
ger_extent = (x_ll, y_ll, x_ur, y_ur)

# Import geoftiff with DEM information
tiff = "/media/nas/satablage/Thomas/Grassdata/srtm_germany_dsm.tif"
params, dem = gtiff.read_geotiff(tiff)
tiffarea = gtiff.tiff2areadef(params['projection'], params['geotransform'],
                              dem.shape)

elevation = image.ImageContainerQuick(dem, tiffarea)

# Directory for cloud physical properties
cpp_dir = '/media/nas/satablage/Thomas/Nebel/CMSAF_microphysics'

# Define time series for analysis
#start = "201408270700"
#end = "201408270715"
start = "201311120830"
end = "201311120845"
#start = "201407110800"
#end = "201407110815"

step = [15]
time_period = get_time_period(start, end, step)

# time_slot = datetime(2014, 8, 27, 7, 15)
for time in time_period:
    print(time)
    # Get cloud microphysic products from CMSAF
    cpp_file = 'CPPin{}00305SVMSG01MD.nc'.format(time.strftime('%Y%m%d%H%M'))
    cpp_filedir = os.path.join(cpp_dir, cpp_file)
    cpp = nc_import(cpp_filedir)
    # Get microphysic paramteers and geo metadata
    proj4 = cpp.getncattr("CMSAF_proj4_params")
    extent = cpp.getncattr("CMSAF_area_extent")
    cot = cpp.variables["cot"][:][0]
    reff = cpp.variables["reff"][:][0]
    cwp = cpp.variables["cwp"][:][0]
    # Geolocate and resample microphysic parameters
    from pyresample import utils
    area_id = 'CPP_cmsaf'
    area_name = 'Gridded cloud physical properties from CMSAF'
    proj_id = 'CPP_cmsaf'
    x_size = cot.shape[0]
    y_size = cot.shape[1]
    cpp_area = utils.get_area_def(area_id, area_name, proj_id, proj4,
                                  x_size, y_size, extent)
    cot_fd = image.ImageContainerQuick(cot, cpp_area)
    reff_fd = image.ImageContainerQuick(reff, cpp_area)
    cwp_fd = image.ImageContainerQuick(cwp, cpp_area)

    # Fog example
    germ_scene = GeostationaryFactory.create_scene(satname="meteosat",
                                                   satnumber='10',
                                                   instrument="seviri",
                                                   time_slot=time)
    germ_scene.load(germ_scene.image.fls_day.prerequisites.add('HRV'),
                    area_extent=ger_extent)
    #germ_scene.project('euro4', mode="nearest")
    #germ_scene.image[0.6].show()

    germ_area = germ_scene[10.8].area_def

    # Resample fls input
    elevation_ger = elevation.resample(germ_area)
    cot_ger = cot_fd.resample(germ_area)
    reff_ger = reff_fd.resample(germ_area)
    cwp_ger = cwp_fd.resample(germ_area)

    ele_img = GeoImage(elevation_ger.image_data, germ_area, time,
                       fill_value=0, mode="L")
    ele_img.enhance(stretch="crude")
    cwp_img = GeoImage(cwp_ger.image_data, germ_area, time,
                       fill_value=0, mode="L")
    cwp_img.enhance(stretch="crude")
    reff_img = GeoImage(reff_ger.image_data, germ_area, time,
                        fill_value=0, mode="L")
    reff_img.enhance(stretch="crude")
    cot_img = GeoImage(cot_ger.image_data, germ_area, time,
                       fill_value=0, mode="L")
    cot_img.enhance(stretch="crude")

    #bgimg = germ_scene[10.8].as_image()
    bgimg = germ_scene.image.ir108()

    fls_img, fogmask = germ_scene.image.fls_day(elevation_ger.image_data,
                                                cot_ger.image_data,
                                                reff_ger.image_data,
                                                cwp_ger.image_data)

    snow_rgb = germ_scene.image.snow()
    daymicro_rgb = germ_scene.image.day_microphysics()
    overview = germ_scene.image.overview()
    # Merge masked and colorized fog clouds with backgrund infrared image
    bgimg.convert("RGB")
    fls_img = Image(fls_img.channels[0], mode='L')
    fls_img.colorize(fogcol)
    fls_img.merge(bgimg)

    ele_img = Image(ele_img.channels[0], mode='L')
    cwp_img = Image(cwp_img.channels[0], mode='L')
    cwp_masked = np.ma.array(cwp_ger.image_data, mask=fogmask)
    print(np.histogram(cwp_masked.compressed()))
    #ele_img.show()
    #cwp_img.show()
    #overview.show()
    #fls_img.show()
    #snow_rgb.show()
    #daymicro_rgb.show()
    ele_img.save("/tmp/fog_example_msg_ger_elevation_{}.png"
                 .format(time.strftime("%Y%m%d%H%M")))
    cot_img.save("/tmp/fog_example_msg_ger_cot_{}.png"
                 .format(time.strftime("%Y%m%d%H%M")))
    reff_img.save("/tmp/fog_example_msg_ger_reff_{}.png"
                  .format(time.strftime("%Y%m%d%H%M")))
    cwp_img.save("/tmp/fog_example_msg_ger_cwp_{}.png"
                 .format(time.strftime("%Y%m%d%H%M")))
    fls_img.save("/tmp/fog_example_msg_ger_fls_{}.png"
                 .format(time.strftime("%Y%m%d%H%M")))
    overview.save("/tmp/fog_example_msg_ger_overview_{}.png"
                  .format(time.strftime("%Y%m%d%H%M")))
    snow_rgb.save("/tmp/fog_example_msg_ger_snow_rgb_{}.png"
                  .format(time.strftime("%Y%m%d%H%M")))
    daymicro_rgb.save("/tmp/fog_example_msg_ger_daymicro_rgb_{}.png"
                      .format(time.strftime("%Y%m%d%H%M")))
    #germ_scene["HRV"].show()

    # Add synoptical mreports from stations
    arrshp = germ_scene[10.8].shape
    bufr_dir = '/data/tleppelt/skydata/'
    bufr_file = "result_{}".format(time.strftime("%Y%m%d"))

    inbufr = os.path.join(bufr_dir, bufr_file)
    from import_synop import read_synop
    stations = read_synop(inbufr, 'visibility')
    currentstations = stations[time.strftime("%Y%m%d%H0000")]
    lats = [i[1] for i in currentstations]
    lons = [i[2] for i in currentstations]
    vis = [i[3] for i in currentstations]

    # Create array for synop parameter
    visarr = np.empty(arrshp)
    visarr.fill(np.nan)
    # Define custom fog colormap
    viscol = Colormap((0., (1.0, 0.0, 0.0)),
                      (5000, (0.7, 0.7, 0.7)))
    vis_colset = Colormap((0, (228 / 255.0, 26 / 255.0, 28 / 255.0)),
                          (1000, (152 / 255.0, 78 / 255.0, 163 / 255.0)),
                          (5000, (55 / 255.0, 126 / 255.0, 184 / 255.0)),
                          (10000, (77 / 255.0, 175 / 255.0, 74 / 255.0)))
    x, y = (germ_area.get_xy_from_lonlat(lons, lats))
    vis_ma = np.ma.array(vis, mask=x.mask)
    visarr[y.compressed(), x.compressed()] = vis_ma.compressed()
    visarr_ma = np.ma.masked_invalid(visarr)
    station_img = Image(visarr_ma, mode='L')
    station_img.colorize(vis_colset)
    station_img.merge(fls_img)
    station_img.save("/tmp/fog_example_msg_ger_fls_stations_{}.png"
                     .format(time.strftime("%Y%m%d%H%M")))
    #fogimg = PILimage.open("/tmp/fog_example_msg_ger_fls_{}.png"
    #                       .format(time.strftime("%Y%m%d%H%M")))
    #for lon, lat in zip(lons, lats):
    #    draw_synop(fogimg, germ_area, lon, lat, 0)
    #fogimg.save("/tmp/fog_example_msg_ger_fls_stations_{}.png"
    #            .format(time.strftime("%Y%m%d%H%M")))
