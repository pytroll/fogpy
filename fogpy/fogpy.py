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

from lowwatercloud import LowWaterCloud
""" This module implements a satellite image based fog and low stratus
detection and forecasting algorithm as a PyTROLL custom composite object.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from mpop.imageo.geo_image import GeoImage
from pyorbital import astronomy
from datetime import datetime
from scipy.ndimage import measurements
from scipy import ndimage
from numpy.lib.stride_tricks import as_strided
from scipy.signal import find_peaks_cwt
from lowwatercloud import LowWaterCloud
from mpop.tools import estimate_cth

logger = logging.getLogger(__name__)


def fls_day(self, elevation, cot, reff, lwp=None):
    """ This method defines a composite for fog and low stratus detection
    and forecasting at daytime.

    Required additional inputs:
        elevation    Ditital elevation model as array
        cot    Cloud optical thickness(depth) as array
        reff    Cloud particle effective radius as array
        lwp    Liquid water path as array
    """
    logger.debug("Creating fog composite for {} instrument"
                 .format(self.fullname))

    self.check_channels(0.635, 0.81, 1.64, 3.92, 8.7, 10.8, 12.0)
    chn108 = self[10.8].data
    chn39 = self[3.92].data
    chn08 = self[0.81].data
    chn16 = self[1.64].data
    chn06 = self[0.635].data
    chn87 = self[8.7].data
    chn120 = self[12.0].data
    time = self.time_slot

    # Get central lon/lat coordinates for the image
    area = self[10.8].area
    lon, lat = area.get_lonlats()

    # Compute fog mask
    fogmask, lcth = fogpy(chn108, chn39, chn08, chn16, chn06, chn87, chn120,
                          time, lat, lon, elevation, cot, reff)
    print(np.sum(fogmask))
    print(lcth.iteritems())
    #img = GeoImage(lcth, self.area, self.time_slot,
    #               fill_value=0, mode="L")
    #img.enhance(stretch="crude")
    #img.show()

    if lwp is not None:
        # Calculate cloud top height
        cth = estimate_cth(chn108, 'midlatitude winter')
        cth = np.ma.filled(cth, np.nan)

        # Apply low cloud mask
        cth_ma = np.ma.array(cth, mask=fogmask)
        # Convert lwp from kg m-2 to g m-2
        lwp_ma = np.ma.array(lwp * 1000, mask=fogmask)
        ctt_ma = np.ma.array(chn108, mask=fogmask)
        print(cth_ma.count())
        # Calibrate cloud base height
        v_get_cloud_base_height = np.vectorize(get_cloud_base_height)
        cbh = v_get_cloud_base_height(lwp_ma.compressed(), cth_ma.compressed(),
                                      ctt_ma.compressed())

    # Apply fog mask to 10.8 channel
    out = np.ones(fogmask.shape)
    out_ma = np.ma.masked_where(fogmask, out)

    # Plot infrared 10.8 um channel with fog mask as image
    img = GeoImage(out_ma, self.area, self.time_slot,
                   fill_value=0, mode="L")
    img.enhance(stretch="crude")

    return(img, fogmask)

fls_day.prerequisites = set([0.635, 0.81, 1.64, 3.92, 8.7, 10.8, 12.0])


def fls_night(self):
    """ This method defines a composite for fog and low stratus detection
    and forecasting at night."""
    pass


def fls_day_modis(self, elevation, cot, reff):
    """ This method defines a composite for fog and low stratus detection
    and forecasting at daytime.

    Required additional inputs:
        elevation    Ditital elevation model as array
        cot    Cloud optical thickness(depth) as array
        reff    cloud particle effective radius as array
    """
    logger.debug("Creating fog composite for {} instrument"
                 .format(self.fullname))
    self.check_channels(0.635, 0.86, 1.64, 3.959, 8.7, 10.8, 12.0)
    chn108 = self[10.8].data
    chn39 = self[3.959].data
    chn08 = self[0.86].data
    chn16 = self[1.64].data
    chn06 = self[0.635].data
    chn87 = self[8.7].data
    chn120 = self[12.0].data

    time = self.time_slot

    # Get central lon/lat coordinates for the image
    area = self[10.8].area
    lon, lat = area.get_lonlats()

    # Compute fog mask
    fogmask = fogpy(chn108, chn39, chn08, chn16, chn06, chn87, chn120, time,
                    lat, lon, elevation, cot, reff)

    # Apply fog mask to 10.8 channel
    out = np.ones(fogmask.shape)
    out_ma = np.ma.masked_where(fogmask, out)

    # Plot infrared 10.8 um channel with fog mask as image
    img = GeoImage(out_ma, self.area, self.time_slot,
                   fill_value=0, mode="L")
    img.enhance(stretch="crude")

    return(img)

fls_day_modis.prerequisites = set([0.635, 0.86, 1.64, 3.959, 8.7, 10.8, 12.0])

# List of composites for SEVIRI instrument
seviri = [fls_day, fls_night]
modis = [fls_day_modis, fls_night]


def fogpy(chn108, chn39, chn08, chn16, chn06, chn87, chn120, time, lat, lon,
          elevation, cot, reff):
    """ The fog and low stratus detection and forecasting algorithms are
    utilizing the methods proposed in different innovative studies:

    Arguements:
        chn108    Array for the 10.8 μm channel
        chn39    Array for the 3.9 μm channel
        chn08    Array for the 0.8 μm channel
        chn16    Array for the 1.6 μm channel
        chn06    Array for the 0.6 μm channel
        chn87    Array for the 8.7 μm channel
        chn120    Array for the 12.0 μm channel
        time    Datetime object for the satellite scence
        lat    Array of latitude values
        lon    Array of longitude values
        elevation Array of area elevation
        cot    Array of cloud optical thickness (depth)
        reff    Array of cloud particle effective raduis

    Returns:
        Infrared image with fog mask

    - A novel approach to fog/low stratus detection using Meteosat 8 data
            J. Cermak & J. Bendix
    - Detecting ground fog from space – a microphysics-based approach
            J. Cermak & J. Bendix

    The algorithm can be applied to satellite zenith angle lower than 70°
    and a maximum solar zenith angle of 80°.

    The algorithm workflow is a succession of differnt masking approaches
    from coarse to finer selection to find fog and low stratus clouds within
    provided satellite images.

            Input: Calibrated satellite images >-----
                                                    |
                1.  Cloud masking -------------------
                                                    |
                2.  Spatial clustering---------------
                                                    |
                3.  Maximum margin elevation --------
                                                    |
                4.  Surface homogenity check --------
                                                    |
                5.  Microphysics plausibility check -
                                                    |
                6.  Differenciate fog - low status --
                                                    |
                7.  Fog dissipation -----------------
                                                    |
                8.  Nowcasting ----------------------
                                                    |
            Output: fog and low stratus mask <--
    """
    arrsize = chn108.size
    # Dictionary of filtered values.
    filters = {}
    prev = 0

    # 1. Cloud masking
    logger.info("### Applying fog cloud filters to input array ###")
    # Given the combination of a solar and a thermal signal at 3.9 μm,
    # the difference in radiances to the 10.8 μm must be larger for a
    # cloud-contaminated pixel than for a clear pixel:
    cm_diff = chn108 - chn39

    # In the histogram of the difference the clear sky peak is identified
    # within a certain range. The nearest significant relative minimum in the
    # histogram towards more negative values is detected and used as a
    # threshold to separate clear from cloudy pixels in the image.

    # Create histogram
    hist = (np.histogram(cm_diff.compressed(), bins='auto'))

    # Find local min and max values
    localmin = (np.diff(np.sign(np.diff(hist[0]))) > 0).nonzero()[0] + 1
    localmax = (np.diff(np.sign(np.diff(hist[0]))) < 0).nonzero()[0] + 1

    # Utilize scipy signal funciton to find peaks
    peakind = find_peaks_cwt(hist[0], np.arange(1, len(hist[1]) / 10))
    peakrange = hist[1][peakind][(hist[1][peakind] >= -10) &
                                 (hist[1][peakind] < 10)]
    minpeak = np.min(peakrange)
    maxpeak = np.max(peakrange)

    # Determine threshold
    logger.debug("Histogram range for cloudy/clear sky pixels: {} - {}"
                 .format(minpeak, maxpeak))

    #plt.bar(hist[1][:-1], hist[0])
    #plt.title("Histogram with 'auto' bins")
    #plt.show()

    thres = np.max(hist[1][localmin[(hist[1][localmin] <= maxpeak) &
                                    (hist[1][localmin] >= minpeak) &
                                    (hist[1][localmin] < 0.5)]])
    if thres > 0 or thres < -5:
        logger.warning("Cloud maks difference threshold {} outside normal"
                       " range (from -5 to 0)".format(thres))
    else:
        logger.debug("Cloud mask difference threshold set to %s" % thres)
    # Create cloud mask for image array
    cloud_mask = cm_diff > thres

    filters['cloud'] = np.nansum(cloud_mask)
    prev += filters['cloud']
    fog_mask = cloud_mask

    # Remove remaining snow pixels
    chn108_ma = np.ma.masked_where(cloud_mask, chn108)
    chn16_ma = np.ma.masked_where(cloud_mask, chn16)
    chn08_ma = np.ma.masked_where(cloud_mask, chn08)
    chn06_ma = np.ma.masked_where(cloud_mask, chn06)

    # Snow has a certain minimum reflectance (0.11 at 0.8 μm) and snow has a
    # certain minimum temperature (256 K)

    # Snow displays a lower reflectivity than water clouds at 1.6 μm, combined
    # with a slightly higher level of absorption (Wiscombe and Warren, 1980)
    # Calculate Normalized Difference Snow Index
    ndsi = (chn06 - chn16) / (chn06 + chn16)

    # Where the NDSI exceeds a certain threshold (0.4) and the two other
    # criteria are met, a pixel is rejected as snow-covered.
    snow_mask = (chn08 / 100 >= 0.11) & (chn108 >= 256) & (ndsi >= 0.4)

    fog_mask = fog_mask | snow_mask
    filters['snow'] = np.nansum(fog_mask) - prev
    prev += filters['snow']

    # Ice cloud exclusion
    # Only warm fog (i.e. clouds in the water phase) are considered.
    # No ice fog!!!
    # Difference of brightness temperatures in the 12.0 and 8.7 μm channels
    # is used as an indicator of cloud phase (Strabala et al., 1994).
    # Where it exceeds 2.5 K, a water-cloud-covered pixel is assumed with a
    # large degree of certainty.
    chn120_ma = np.ma.masked_where(cloud_mask | snow_mask, chn120)
    chn108_ma = np.ma.masked_where(cloud_mask | snow_mask, chn108)
    chn87_ma = np.ma.masked_where(cloud_mask | snow_mask, chn87)
    ic_diff = chn120 - chn87

    # Straightforward temperature test, cutting off at very low 10.8 μm
    # brightness temperatures (250 K).
    # Create ice cloud mask
    ice_mask = (ic_diff < 2.5) | (chn108 < 250)

    fog_mask = fog_mask | ice_mask
    filters['ice'] = np.nansum(fog_mask) - prev
    prev += filters['ice']

    # Thin cirrus is detected by means of the split-window IR channel
    # brightness temperature difference (T10.8 –T12.0 ). This difference is
    # compared to a threshold dynamically interpolated from a lookup table
    # based on satellite zenith angle and brightness temperature at 10.8 μm
    # (Saunders and Kriebel, 1988)
    chn120_ma = np.ma.masked_where(cloud_mask | snow_mask | ice_mask, chn120)
    chn108_ma = np.ma.masked_where(cloud_mask | snow_mask | ice_mask, chn108)

    bt_diff = chn108 - chn120

    # Calculate sun zenith angles
    sza = astronomy.sun_zenith_angle(time, lon, lat)

    minsza = np.min(sza)
    maxsza = np.max(sza)
    logger.debug("Found solar zenith angles from %s to %s°" % (minsza,
                                                               maxsza))

    # Calculate secant of sza
    # secsza = np.ma.masked_where(cloud_mask | snow_mask | ice_mask,
    #                             (1 / np.cos(np.deg2rad(sza))))
    secsza = 1 / np.cos(np.deg2rad(sza))

    # Lookup table for BT difference thresholds at certain sec(sun zenith
    # angles) and 10.8 μm BT
    lut = {260: {1.0: 0.55, 1.25: 0.60, 1.50: 0.65, 1.75: 0.90, 2.0: 1.10},
           270: {1.0: 0.58, 1.25: 0.63, 1.50: 0.81, 1.75: 1.03, 2.0: 1.13},
           280: {1.0: 1.30, 1.25: 1.61, 1.50: 1.88, 1.75: 2.14, 2.0: 2.30},
           290: {1.0: 3.06, 1.25: 3.72, 1.50: 3.95, 1.75: 4.27, 2.0: 4.73},
           300: {1.0: 5.77, 1.25: 6.92, 1.50: 7.00, 1.75: 7.42, 2.0: 8.43},
           310: {1.0: 9.41, 1.25: 11.22, 1.50: 11.03, 1.75: 11.60, 2.0: 13.39}}

    # Apply lut to BT and sza values

    def find_nearest_lut_sza(sza):
        """ Get nearest look up table key value for given ssec(sza)"""
        sza_opt = [1.0, 1.25, 1.50, 1.75, 2.0]
        sza_idx = np.array([np.abs(sza - i) for i in sza_opt]).argmin()
        return(sza_opt[sza_idx])

    def find_nearest_lut_bt(bt):
        """ Get nearest look up table key value for given BT"""
        bt_opt = [260, 270, 280, 290, 300, 310]
        bt_idx = np.array([np.abs(bt - i) for i in bt_opt]).argmin()
        return(bt_opt[bt_idx])

    def apply_lut(sza, bt):
        """ Apply LUT to given BT and sza values"""
        return(lut[bt][sza])

    # Vectorize LUT functions for numpy arrays
    vfind_nearest_lut_sza = np.vectorize(find_nearest_lut_sza)
    vfind_nearest_lut_bt = np.vectorize(find_nearest_lut_bt)
    vapply_lut = np.vectorize(apply_lut)

    secsza_lut = vfind_nearest_lut_sza(secsza)
    chn108_ma_lut = vfind_nearest_lut_bt(chn108)

    bt_thres = vapply_lut(secsza_lut, chn108_ma_lut)
    logger.debug("Set BT difference threshold for thin cirrus from %s to %s K"
                 % (np.min(bt_thres), np.max(bt_thres)))
    # Create thin cirrus mask
    bt_ci_mask = bt_diff > bt_thres

    # Other cirrus test (T8.7–T10.8), founded on the relatively strong cirrus
    # signal at the former wavelength (Wiegner et al.1998). Where the
    # difference is greater than 0 K, cirrus is assumed to be present.
    strong_ci_diff = chn87 - chn108
    strong_ci_mask = strong_ci_diff > 0
    cirrus_mask = bt_ci_mask | strong_ci_mask

    fog_mask = fog_mask | cirrus_mask
    filters['cirrus'] = np.nansum(fog_mask) - prev
    prev += filters['cirrus']

    # Those pixels whose cloud phase still remains undefined after these ice
    # cloud exclusions are subjected to a much weaker cloud phase test in order
    # to get an estimate regarding their phase. This test uses the NDSI
    # introduced above. Where it falls below 0.1, a water cloud is assumed to
    # be present.
    chn16_ma = np.ma.masked_where(cloud_mask | snow_mask | ice_mask |
                                  cirrus_mask, chn16)
    chn06_ma = np.ma.masked_where(cloud_mask | snow_mask | ice_mask |
                                  cirrus_mask, chn06)
    ndsi_ci = (chn06 - chn16) / (chn06 + chn16)
    water_mask = ndsi_ci > 0.1

    fog_mask = fog_mask | water_mask
    filters['water'] = np.nansum(fog_mask) - prev
    prev += filters['water']

    # Small droplet proxy test
    # Fog generally has a stronger signal at 3.9 μm than clear ground, which
    # in turn radiates more than other clouds.
    # The 3.9 μm radiances for cloud-free land areas are averaged over 50 rows
    # at a time to obtain an approximately latitudinal value.
    # Wherever a cloud-covered pixel exceeds this value, it is flagged
    # ‘small droplet cloud’.
    cloud_free_ma = np.ma.masked_where(~cloud_mask, chn39)
    chn39_ma = np.ma.masked_where(cloud_mask | snow_mask | ice_mask |
                                  cirrus_mask | water_mask, chn39)
    # Latitudinal average cloud free radiances
    lat_cloudfree = np.ma.mean(cloud_free_ma, 1)
    logger.debug("Mean latitudinal threshold for cloudfree areas: %.2f K"
                 % np.mean(lat_cloudfree))
    global line
    line = 0

    def find_watercloud(lat, thres):
        """Funciton to compare row of BT with given latitudinal thresholds"""
        global line
        if all(lat.mask):
            res = lat.mask
        elif np.ma.is_masked(thres[line]):
            res = lat <= np.mean(lat_cloudfree)
        else:
            res = lat <= thres[line]
        line += 1

        return(res)

    # Apply latitudinal threshold to cloudy areas
    drop_mask = np.apply_along_axis(find_watercloud, 1, chn39,
                                    lat_cloudfree)

    fog_mask = fog_mask | drop_mask
    filters['drop'] = np.nansum(fog_mask) - prev
    prev += filters['drop']

    # Apply previous defined filters
    chn108_ma = np.ma.masked_where(fog_mask, chn108)

    logger.debug("Number of filtered non-fog pixels: %s"
                 % (np.sum(chn108_ma.mask)))
    logger.debug("Number of potential fog cloud pixels: %s"
                 % (np.sum(~chn108_ma.mask)))

    # 2. Spatial clustering
    logger.info("### Analizing spatial properties of filtered fog clouds ###")

    # Enumerate fog cloud clusters
    cluster = measurements.label(~chn108_ma.mask)

    # Get 10.8 channel sampled by the previous fog filters
    cluster_ma = np.ma.masked_where(fog_mask, cluster[0])
    # Get cloud and snow free cells
    clear_ma = np.ma.masked_where(~cloud_mask | snow_mask, chn108)

    logger.debug("Number of spatial coherent fog cloud clusters: %s"
                 % np.nanmax(np.unique(cluster_ma)))

    # 3. Altitude test
    def sliding_window(arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1,
                 arr.shape[1] - window_size + 1,
                 window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
                   arr.shape[1]*arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

    def cell_neighbors(arr, i, j, d, value):
        """Return d-th neighbors of cell (i, j)"""
        w = sliding_window(arr, 2*d+1)

        ix = np.clip(i - d, 0, w.shape[0]-1)
        jx = np.clip(j - d, 0, w.shape[1]-1)

        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)

        # Get cell value
        if i1 - i0 == 3:
            icell = 1
        elif (i1 - i0 == 2) & (i0 == 0):
            icell = 0
        elif (i1 - i0 == 2) & (i0 == 1):
            icell = 2
        if j1 - j0 == 3:
            jcell = 1
        elif (j1 - j0 == 2) & (j0 == 0):
            jcell = 0
        elif (j1 - j0 == 2) & (j0 == 1):
            jcell = 2

        irange = range(i0, i1)
        jrange = range(j0, j1)
        neighbors = [w[ix, jx][k, l] for k in irange for l in jrange
                     if k != icell or l != jcell]
        center = value[i, j]  # Get center cell value from additional array

        return(center, neighbors)

    def get_fog_cth(cluster, cf_arr, bt_cc, elevation):
        """Get neighboring cloud free BT and elevation values of potenital
        fog cloud clusters and compute cloud top height from naximum BT
        differences for fog cloud contaminated pixel in comparison to cloud
        free areas and their corresponding elevation
        """
        e = 1
        from collections import defaultdict
        result = defaultdict(list)
        elevation_ma = np.ma.masked_where(~cloud_mask | snow_mask, elevation)
        # Convert masked values to nan
        if np.ma.isMaskedArray(cf_arr):
            cf_arr = cf_arr.filled(np.nan)
        for index, val in np.ndenumerate(cluster):
            if val != 0:
                # Get list of cloud free neighbor pixel
                tcc, tneigh = cell_neighbors(cf_arr, *index, d=1,
                                             value=bt_cc)
                zcc, zneigh = cell_neighbors(elevation_ma, *index, d=1,
                                             value=elevation)
                tcf_diff = np.array([tcf - tcc for tcf in tneigh])
                zcf_diff = np.array([zcf - zcc for zcf in zneigh])
                # Get maximum bt difference
                try:
                    maxd = np.nanargmax(tcf_diff)
                except ValueError:
                    continue
                # compute cloud top height with constant athmospere temperature
                # lapse rate
                rate = 0.65
                cth = tcf_diff[maxd] / rate * 100 - zcf_diff[maxd]
                result[val].append(cth)

        return(result)
    # Calculate fog cluster cloud top height
    cluster_h = get_fog_cth(cluster_ma, clear_ma, chn108_ma, elevation)

    # Apply maximum threshold for cluster height to identify low fog clouds
    cluster_mask = cluster_ma.mask
    for key, item in cluster_h.iteritems():
        if any([c > 2000 for c in item]):
            cluster_mask[cluster_ma[key]] = True

    # Create additional fog cluster map
    cluster_cth = np.ma.masked_where(cluster_mask, cluster_ma)
    for key, item in cluster_h.iteritems():
        if all([c <= 2000 for c in item]):
            cluster_cth[cluster_cth[key]] = np.mean(item)

    # Update fog filter
    fog_mask = fog_mask | cluster_mask
    filters['height'] = np.nansum(fog_mask) - prev
    prev += filters['height']

    # Apply previous defined spatial filters
    chn108_ma = np.ma.masked_where(cluster_mask, chn108)

    # Surface homogenity test
    cluster, nlbl = ndimage.label(~chn108_ma.mask)
    cluster_ma = np.ma.masked_where(cluster_mask, cluster)

    cluster_sd = ndimage.standard_deviation(chn108_ma, cluster_ma,
                                            index=np.arange(1, nlbl+1))

    # 4. Mask potential fog clouds with high spatial inhomogenity
    sd_mask = cluster_sd > 2.5
    cluster_dict = {key: sd_mask[key - 1] for key in np.arange(1, nlbl+1)}
    for val in np.arange(1, nlbl+1):
        cluster_mask[cluster_ma == val] = cluster_dict[val]

    fog_mask = fog_mask | cluster_mask
    filters['homogen'] = np.nansum(fog_mask) - prev
    prev += filters['homogen']

    # Apply previous defined spatial filters
    chn108_ma = np.ma.masked_where(cluster_mask, chn108)

    logger.debug("Number of spatial filtered non-fog pixels: %s"
                 % (np.sum(chn108_ma.mask)))
    logger.debug("Number of remaining fog cloud pixels: %s"
                 % (np.sum(~chn108_ma.mask)))

    # 5. Apply microphysical fog cloud filters
    # Typical microphysical parameters for fog were taken from previous studies
    # Fog optical depth normally ranges between 0.15 and 30 while droplet
    # effective radius varies between 3 and 12 μm, with a maximum of 20 μm in
    # coastal fog. The respective maxima for optical depth (30) and droplet
    # radius (20 μm) are applied to the low stratus mask as cut-off levels.
    # Where a pixel previously identified as fog/low stratus falls outside the
    # range it will now be flagged as a non-fog pixel.
    logger.info("### Apply microphysical plausible check ###")

    if np.ma.isMaskedArray(cot):
        cot = cot.base
    if np.ma.isMaskedArray(reff):
        reff = reff.base

    # Add mask by microphysical thresholds
    cpp_mask = cluster_mask | (cot > 30) | (reff > 20e-6)

    fog_mask = fog_mask | cpp_mask
    filters['cpp'] = np.nansum(fog_mask) - prev
    prev += filters['cpp']

    # Apply previous defined microphysical filters
    chn108_ma = np.ma.masked_where(cpp_mask, chn108)

    logger.debug("Number of microphysical filtered non-fog pixels: %s"
                 % (np.sum(chn108_ma.mask)))
    logger.debug("Number of remaining fog cloud pixels: %s"
                 % (np.sum(~chn108_ma.mask)))

    # Combine single masks to derive potential fog cloud filter
    fls_mask = (fog_mask)

    # Create debug output
    filters['fls'] = np.nansum(fog_mask)
    filters['remain'] = np.nansum(~fog_mask)

    logger.info("""---- FLS algorithm filter results ---- \n
    Number of initial pixels:            {}
    Removed non cloud pixel:             {}
    Removed snow pixels:                 {}
    Removed ice cloud pixels:            {}
    Removed thin cirrus pixels:          {}
    Removed non water cloud pixels       {}
    Removed non small droplet pixels     {}
    Removed spatial high cloud pixels    {}
    Removed spatial inhomogen pixels     {}
    Removed microphysical tested pixels  {}
    ---------------------------------------
    Filtered non fog/low stratus pixels  {}
    Remaining fog/low stratus pixels     {}
    ---------------------------------------
    """.format(arrsize, filters['cloud'], filters['snow'],
               filters['ice'], filters['cirrus'], filters['water'],
               filters['drop'], filters['height'], filters['homogen'],
               filters['cpp'], filters['fls'], filters['remain']))

    return(fog_mask, cluster_h)


def get_cloud_base_height(lwp, cth, ctt):
    """ Calculate cloud base heights for low cloud pixels with a
    numerical 1-D low cloud model and known liquid water path, cloud top height
    and temperature from satellite retrievals"""

    lowcloud = LowWaterCloud(cth, ctt, lwp)
    cbh = lowcloud.optimize_cbh(0., method='basin')

    return(cbh)
