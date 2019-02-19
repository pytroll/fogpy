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

"""This module implements an basic algorithm filter class
and several class instances for satellite fog detection applications"""

import copyreg
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import types

from copy import deepcopy
from collections import defaultdict
from datetime import datetime
from matplotlib.cm import get_cmap
import multiprocessing as mp
from pyorbital import astronomy
from scipy.signal import find_peaks_cwt
from scipy import ndimage
from fogpy.lowwatercloud import LowWaterCloud
from utils.import_synop import read_synop

logger = logging.getLogger(__name__)


# Add new pickle method, required for multiprocessing to run class instance
# methods in parallel
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)


class NotApplicableError(Exception):
    """Exception to be raised when a filter is not applicable."""
    pass


class BaseArrayFilter(object):
    """This super filter class provide all functionalities to apply a filter
    funciton on a given numpy array representing a satellite image and return
    the filtered masked array as result."""
    def __init__(self, arr, **kwargs):
        if isinstance(arr, np.ma.MaskedArray):
            self.arr = arr
            self.inmask = arr.mask
        elif isinstance(arr, np.ndarray):
            self.arr = arr
            self.inmask = np.full(arr.shape, False, dtype=bool)
        else:
            raise ImportError('The filter <{}> needs a valid 2d numpy array '
                              'as input'.format(self.__class__.__name__))
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                self.__setattr__(key, value)
            self.result = None
            self.mask = None
        # Get class name
        self.name = self.__str__().split(' ')[0].split('.')[-1]
        # Set time
        if not hasattr(self, 'time'):
            self.time = datetime.now()
            logger.debug('Setting filter reference time to current time: {}'
                         .format(self.time))
        # Set plotting attribute
        if not hasattr(self, 'save'):
            self.save = False
        if not hasattr(self, 'plot'):
            self.plot = False
        if not hasattr(self, 'dir'):
            self.dir = '/tmp'
        if not hasattr(self, 'bg_img'):
            self.bg_img = self.arr
        if not hasattr(self, 'resize'):
            self.resize = 0
        # Get number of cores
        if not hasattr(self, 'nprocs'):
            self.nprocs = mp.cpu_count()
        # Get attribute names for plotting
        if not hasattr(self, 'plotattr'):
            self.plotattr = None

    @property
    def mask(self):
        """Filter mask getter method."""
        return self._mask

    @mask.setter
    def mask(self, value):
        """Filter mask setter method."""
        if value is None:
            self._mask = value
        elif self.inmask.ndim != value.ndim:
            logger.info("Mask dimensions differ. Reshaping to input shape: {}"
                        .format(self.inmask.shape))
            newvalue = np.reshape(value, self.inmask.shape)
            self._mask = newvalue
        else:
            self._mask = value

    def apply(self):
        """Apply the given filter function."""
        if self.isapplicable():
            self.filter_function()
            self.check_results()
        else:
            raise NotApplicableError('Array filter <{}> is not applicable'
                                     .format(self.__class__.__name__))

        return self.result, self.mask

    def isapplicable(self):
        """Test filter applicability."""
        ret = []
        for attr in self.attrlist:
            if hasattr(self, attr):
                ret.append(True)
            else:
                ret.append(False)
                logger.warning("Missing input attribute: {}".format(attr))

        return all(ret)

    def filter_function(self):
        """Filter routine"""
        self.mask = np.ones(self.arr.shape) == 1

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def check_results(self):
        """Check filter results for plausible results."""
        self.filter_stats()
        if self.plot:
            self.plot_filter(self.save, self.dir, self.resize,
                             attr=self.plotattr)
        ret = True
        return ret

    def filter_stats(self):
        self.filter_size = self.mask.size
        self.filter_num = np.nansum(self.mask)
        if self.inmask is None:
            self.inmask_num = 0
            self.new_masked = self.filter_num
            self.remain_num = self.filter_size - self.filter_num
        else:
            self.inmask_num = np.nansum(self.inmask)
            self.new_masked = np.nansum(~self.inmask & self.mask)
            self.remain_num = np.nansum(~self.mask & ~self.inmask)

        logger.info("""Filter results for {} \n
                    {}
                    Array size:              {}
                    Masking:                 {}
                    Previous masked:         {}
                    New filtered:            {}
                    Remaining:               {}"""
                    .format(self.name, self.__doc__,
                            self.filter_size, self.filter_num, self.inmask_num,
                            self.new_masked, self.remain_num))

    def plot_filter(self, save=False, dir="/tmp", resize=0, attr=None,
                    type='png', area=None, name=None):
        """Plotting the filter result.

        .. Note:: Masks should be correctly setup to be plotted:
                  **True** (1) mask values are not shown, **False** (0) mask
                  values are displayed.
        """
        if name is None:
            name = self.name
        # Get output directory and image name
        savedir = os.path.join(dir, name + '_' +
                               datetime.strftime(self.time,
                                                 '%Y%m%d%H%M') + '.' + type)
        maskdir = os.path.join(dir, name + '_mask_' +
                               datetime.strftime(self.time,
                                                 '%Y%m%d%H%M') + '.' + type)
        # Using Trollimage if available, else matplotlib is used to plot
        try:
            from trollimage.image import Image
            from trollimage.colormap import Colormap
            from mpop.imageo.geo_image import GeoImage
        except:
            cmap = get_cmap('gray')
            cmap.set_bad('goldenrod', 1.)
            imgplot = plt.imshow(self.result.squeeze())
            plt.axis('off')
            plt.tight_layout()
            if save:
                plt.savefig(savedir, bbox_inches='tight', pad_inches=0.0)
                logger.info("{} results are plotted to: {}". format(self.name,
                                                                    self.dir))
            else:
                plt.show()
        # check area attribute
        if area is None and type == 'tif':
            try:
                area = self.area
            except:
                Warning("Area object not found. Plotting filter result as"
                        " image")
                type = 'png'
        # Define custom fog colormap
        fogcol = Colormap((0., (250 / 255.0, 200 / 255.0, 40 / 255.0)),
                          (1., (1.0, 1.0, 229 / 255.0)))
        maskcol = Colormap((1., (250 / 255.0, 200 / 255.0, 40 / 255.0)))

        # Create image from data
        if self.result is None:
            self.result = self.arr
        filter_img = Image(self.result.squeeze(), mode='L', fill_value=None)
        filter_img.stretch("crude")
        filter_img.invert()
        filter_img.colorize(fogcol)
        # Get background image
        bg_img = Image(self.bg_img.squeeze(), mode='L', fill_value=None)
        bg_img.stretch("crude")
        bg_img.convert("RGB")
        bg_img.invert()
        if resize != 0:
            if not isinstance(resize, int):
                resize = int(resize)
            bg_img.resize((self.bg_img.shape[0] * resize,
                           self.bg_img.shape[1] * resize))
            filter_img.resize((self.result.shape[0] * resize,
                               self.result.shape[1] * resize))

        try:
            # Merging
            filter_img.merge(bg_img)
        except:
            logger.warning("No merging for filter plot possible")
        if save:
            if type == 'tif':
                result_img = GeoImage(filter_img.channels, area,
                                      self.time, fill_value=None,
                                      mode="RGB")
                result_img.save(savedir)
            else:
                filter_img.save(savedir)
            logger.info("{} results are plotted to: {}". format(self.name,
                                                                savedir))
        else:
            filter_img.show()
        # Create mask image
        if isinstance(self.result, np.ma.masked_array):
            mask = self.result.mask.squeeze()
            mask = np.ma.masked_where(mask == 1, mask)
            mask_img = Image(mask, mode='L', fill_value=None)
            mask_img.stretch("crude")
            mask_img.invert()
            mask_img.colorize(maskcol)
            if resize != 0:
                if not isinstance(resize, int):
                    resize = int(resize)
                mask_img.resize((self.result.shape[0] * resize,
                                 self.result.shape[1] * resize))
            # mask_img.merge(bg_img)
            if save:
                if type == 'tif':
                    result_img = GeoImage(mask_img.channels, area,
                                          self.time, fill_value=None,
                                          mode="RGB")
                    result_img.save(maskdir)
                else:
                    mask_img.save(maskdir)
            else:
                mask_img.show()
        # Create optional attribute images
        if attr is not None:
            if isinstance(attr, list):
                for a in attr:
                    self._plot_image(a, save, dir, resize, type, area)
            elif isinstance(attr, str):
                self._plot_image(attr, save, dir, resize, type, area)

    def _plot_image(self, name, save=False, dir="/tmp", resize=0, type='png',
                    area=None):
        """Plotting function for additional filter attributes.

        .. Note:: Masks should be correctly setup to be plotted:
                  **True** (1) mask values are not shown, **False** (0) mask
                  values are displayed.
        """
        from trollimage.image import Image
        from trollimage.colormap import Colormap
        # Define custom fog colormap
        fogcol = Colormap((0., (250 / 255.0, 200 / 255.0, 40 / 255.0)),
                          (1., (1.0, 1.0, 229 / 255.0)))
        maskcol = Colormap((1., (250 / 255.0, 200 / 255.0, 40 / 255.0)))
        # Get save directory
        attrdir = os.path.join(dir, self.name + '_' + name + '_' +
                               datetime.strftime(self.time,
                                                 '%Y%m%d%H%M') + '.' + type)
        logger.info("Plotting filter attribute {} to {}".format(name, attrdir))
        # Generate image
        attr = getattr(self, name)
        if attr.dtype == 'bool':
            attr = np.ma.masked_where(attr == 1, attr)
        attr_img = Image(attr.squeeze(), mode='L', fill_value=None)
        attr_img.colorize(fogcol)
        # Get background image
        bg_img = Image(self.bg_img.squeeze(), mode='L', fill_value=None)
        bg_img.stretch("crude")
        bg_img.convert("RGB")
        bg_img.invert()
        if resize != 0:
            if not isinstance(resize, int):
                resize = int(resize)
            bg_img.resize((self.bg_img.shape[0] * resize,
                           self.bg_img.shape[1] * resize))
            attr_img.resize((self.result.shape[0] * resize,
                             self.result.shape[1] * resize))
        try:
            # Merging
            attr_img.merge(bg_img)
        except:
            logger.warning("No merging for attribute plot possible")
        if save:
            if type == 'tif':
                from mpop.imageo.geo_image import GeoImage
                result_img = GeoImage(attr_img.channels, area,
                                      self.time, fill_value=None,
                                      mode="RGB")
                result_img.save(attrdir)
            else:
                attr_img.save(attrdir)
        else:
            attr_img.show()

        return(attr_img)


class CloudFilter(BaseArrayFilter):
    """Cloud filtering for satellite images."""
    # Required inputs
    attrlist = ['ir108', 'ir039']

    def filter_function(self):
        """Cloud filter routine

        Given the combination of a solar and a thermal signal at 3.9 μm,
        the difference in radiances to the 10.8 μm must be larger for a
        cloud-contaminated pixel than for a clear pixel.
        In the histogram of the difference the clear sky peak is identified
        within a certain range. The nearest significant relative minimum in the
        histogram towards more negative values is detected and used as a
        threshold to separate clear from cloudy pixels in the image.

        Args:
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | ir039 (:obj:`ndarray`): Array for the 3.9 μm channel.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Cloud Filter")

        # Set cloud confidence range
        if not hasattr(self, 'ccr'):
            self.ccr = 5  # Kelvin
        # Set peak ranges
        if not hasattr(self, 'prange'):
            self.prange = (-20, 10)  # Min - max peak range

        # Infrared channel difference
        self.cm_diff = np.ma.asarray(self.ir108 - self.ir039)

        # Create histogram
        self.hist = (np.histogram(self.cm_diff.compressed(), bins='auto'))

        # Find local min and max values
        peaks = np.sign(np.diff(self.hist[0]))
        localmin = (np.diff(peaks) > 0).nonzero()[0] + 1
        localmax = (np.diff(peaks) < 0).nonzero()[0] + 1

        # Utilize scipy signal funciton to find peaks
        peakind = find_peaks_cwt(self.hist[0],
                                 np.arange(1, len(self.hist[1]) / 10))
        histpeaks = self.hist[1][peakind]
        peakrange = histpeaks[(histpeaks >= self.prange[0]) &
                              (histpeaks < self.prange[1])]
        if len(peakrange) == 1:
            logger.error("Not enough peaks found in range {} - {} \n"
                         "Using slope declination as threshold"
                         .format(self.prange[0], self.prange[1]))
            slope, thres = self.get_slope_decline(self.hist[0],
                                                  self.hist[1][:-1])
            self.thres = thres
        elif len(peakrange) >= 2:
            self.minpeak = np.min(peakrange)
            self.maxpeak = np.max(peakrange)

            # Determine threshold
            logger.debug("Histogram range for cloudy/clear sky pixels: {} - {}"
                         .format(self.minpeak, self.maxpeak))
            thres_index = localmin[(self.hist[1][localmin] <= self.maxpeak) &
                                   (self.hist[1][localmin] >= self.minpeak) &
                                   (self.hist[1][localmin] < 0.5)]
            self.thres = np.max(self.hist[1][thres_index])
        else:
            raise ValueError

        if self.thres > 0 or self.thres < -5:
            logger.warning("Cloud maks difference threshold {} outside normal"
                           " range (from -5 to 0)".format(self.thres))
        else:
            logger.debug("Cloud mask difference threshold set to {}"
                         .format(self.thres))
        # Compute cloud confidence level
        self.ccl = (self.cm_diff - self.thres - self.ccr) / (-2 * self.ccr)
        # Limit range to 0 (cloudfree) and 1 (cloudy)
        self.ccl[self.ccl > 1] = 1
        self.ccl[self.ccl < 0] = 0

        # Create cloud mask for image array
        self.mask = self.cm_diff > self.thres

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def plot_cloud_hist(self, saveto=None):
        """Plot the histogram of brightness temperature differences."""
        plt.bar(self.hist[1][:-1], self.hist[0])
        plt.title("Histogram with 'auto' bins")
        if saveto is None:
            plt.show()
        else:
            plt.savefig(saveto)

    def get_slope_decline(self, y, x):
        """ Compute the slope declination of a histogram."""
        slope = np.diff(y) / np.diff(x)
        decline = slope[1:] * slope[:-1]
        # Point of slope declination
        thres_id = np.where(np.logical_and(slope[1:] > 0, decline > 0))[0]
        if len(thres_id) != 0:
            decline_points = x[thres_id + 2]
            thres = np.min(decline_points[decline_points > -5])
        else:
            thres = None
        return(slope, thres)


class SnowFilter(BaseArrayFilter):
    """Snow filtering for satellite images."""
    # Required inputs
    attrlist = ['vis006', 'vis008', 'nir016', 'ir108']

    def filter_function(self):
        """Snow filter routine

        Snow has a certain minimum reflectance (0.11 at 0.8 μm) and snow has a
        certain minimum temperature (256 K)
        Snow displays a lower reflectivity than water clouds at 1.6 μm,
        combined with a slightly higher level of absorption
        (Wiscombe and Warren, 1980)
        thresholds are applied in combination with the Normalized Difference
        Snow Index.

        Args:
            | vis006 (:obj:`ndarray`): Array for the 0.6 μm channel.
            | nir016 (:obj:`ndarray`): Array for the 1.6 μm channel.
            | vis008 (:obj:`ndarray`): Array for the 0.8 μm channel.
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Snow Filter")
        # Calculate Normalized Difference Snow Index
        self.ndsi = (self.vis006 - self.nir016) / (self.vis006 + self.nir016)

        # Where the NDSI exceeds a certain threshold (0.4) and the two other
        # criteria are met, a pixel is rejected as snow-covered.
        # Create snow mask for image array
        temp_thres = (self.vis008 / 100 >= 0.11) & (self.ir108 >= 256)
        ndsi_thres = self.ndsi >= 0.4
        # Create snow mask for image array
        self.mask = temp_thres & ndsi_thres

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class IceCloudFilter(BaseArrayFilter):
    """Ice cloud filtering for satellite images."""
    # Required inputs
    attrlist = ['ir120', 'ir087', 'ir108']

    def filter_function(self):
        """Ice cloud filter routine

        Difference of brightness temperatures in the 12.0 and 8.7 μm channels
        is used as an indicator of cloud phase (Strabala et al., 1994).
        Where it exceeds 2.5 K, a water-cloud-covered pixel is assumed with a
        large degree of certainty. This is combined with a straightforward
        temperature test, cutting off at very low 10.8 μm brightness
        temperatures (250 K).

        Args:
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | ir087 (:obj:`ndarray`): Array for the 8.7 μm channel.
            | ir120 (:obj:`ndarray`): Array for the 12.0 μm channel.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Snow Filter")
        # Apply infrared channel difference
        self.ic_diff = self.ir120 - self.ir087
        # Create ice cloud mask
        ice_mask = (self.ic_diff < 2.5) | (self.ir108 < 250)
        # Create snow mask for image array
        self.mask = ice_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class CirrusCloudFilter(BaseArrayFilter):
    """Thin cirrus cloud filtering for satellite images."""
    # Required inputs
    attrlist = ['ir120', 'ir087', 'ir108', 'lat', 'lon', 'time']

    def filter_function(self):
        """Ice cloud filter routine

        Thin cirrus is detected by means of the split-window IR channel
        brightness temperature difference (T10.8 –T12.0 ). This difference is
        compared to a threshold dynamically interpolated from a lookup table
        based on satellite zenith angle and brightness temperature at 10.8 μm
        (Saunders and Kriebel, 1988)
        In addtion a second strong cirrus test (T8.7–T10.8), founded on the
        relatively strong cirrus signal at the former wavelength is applied
        (Wiegner et al.1998). Where the difference is greater than 0 K, cirrus
        is assumed to be present.

        Args:
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | ir087 (:obj:`ndarray`): Array for the 8.7 μm channel.
            | ir120 (:obj:`ndarray`): Array for the 12.0 μm channel.
            | time (:obj:`datetime`): Datetime object for the satellite scence.
            | lat (:obj:`ndarray`): Array of latitude values.
            | lon (:obj:`ndarray`): Array of longitude values.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Cirrus Filter")
        # Get infrared channel difference
        self.bt_diff = self.ir108 - self.ir120
        # Calculate sun zenith angles
        sza = astronomy.sun_zenith_angle(self.time, self.lon, self.lat)
        minsza = np.min(sza)
        maxsza = np.max(sza)
        logger.debug("Found solar zenith angles from %s to %s°" % (minsza,
                                                                   maxsza))
        # Calculate secant of sza
        secsza = 1 / np.cos(np.deg2rad(sza))

        # Apply lut to BT and sza values
        # Vectorize LUT functions for numpy arrays
        vfind_nearest_lut_sza = np.vectorize(self.find_nearest_lut_sza)
        vfind_nearest_lut_bt = np.vectorize(self.find_nearest_lut_bt)
        vapply_lut = np.vectorize(self.apply_lut)

        secsza_lut = vfind_nearest_lut_sza(secsza)
        chn108_ma_lut = vfind_nearest_lut_bt(self.ir108)

        self.bt_thres = vapply_lut(secsza_lut, chn108_ma_lut)
        logger.debug("Set BT difference thresholds for cirrus: {} to {} K"
                     .format(np.min(self.bt_thres), np.max(self.bt_thres)))
        # Create thin cirrus mask
        self.bt_ci_mask = self.bt_diff > self.bt_thres

        # Strong cirrus test
        self.strong_ci_diff = self.ir087 - self.ir108
        self.strong_ci_mask = self.strong_ci_diff > 0
        cirrus_mask = self.bt_ci_mask | self.strong_ci_mask

        # Create snow mask for image array
        self.mask = cirrus_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def find_nearest_lut_sza(self, sza):
        """ Get nearest look up table key value for given ssec(sza)."""
        sza_opt = [1.0, 1.25, 1.50, 1.75, 2.0]
        sza_idx = np.array([np.abs(sza - i) for i in sza_opt]).argmin()
        return(sza_opt[sza_idx])

    def find_nearest_lut_bt(self, bt):
        """ Get nearest look up table key value for given BT."""
        bt_opt = [260, 270, 280, 290, 300, 310]
        bt_idx = np.array([np.abs(bt - i) for i in bt_opt]).argmin()
        return(bt_opt[bt_idx])

    def apply_lut(self, sza, bt):
        """ Apply LUT to given BT and sza values."""
        return(self.lut[bt][sza])
    # Lookup table for BT difference thresholds at certain sec(sun zenith
    # angles) and 10.8 μm BT
    lut = {260: {1.0: 0.55, 1.25: 0.60, 1.50: 0.65, 1.75: 0.90, 2.0: 1.10},
           270: {1.0: 0.58, 1.25: 0.63, 1.50: 0.81, 1.75: 1.03, 2.0: 1.13},
           280: {1.0: 1.30, 1.25: 1.61, 1.50: 1.88, 1.75: 2.14, 2.0: 2.30},
           290: {1.0: 3.06, 1.25: 3.72, 1.50: 3.95, 1.75: 4.27, 2.0: 4.73},
           300: {1.0: 5.77, 1.25: 6.92, 1.50: 7.00, 1.75: 7.42, 2.0: 8.43},
           310: {1.0: 9.41, 1.25: 11.22, 1.50: 11.03, 1.75: 11.60, 2.0: 13.39}}


class WaterCloudFilter(BaseArrayFilter):
    """Water cloud filtering for satellite images."""
    # Required inputs
    attrlist = ['vis006', 'nir016', 'ir039', 'cloudmask']

    def filter_function(self):
        """Water cloud filter routine

        Apply a weaker cloud phase test in order to get an estimate regarding
        their phase. This test uses the NDSI introduced above. Where it falls
        below 0.1, a water cloud is assumed to be present.
        Afterwards a small droplet proxy tes is being performed. Fog generally
        has a stronger signal at 3.9 μm than clear ground, which in turn
        radiates more than other clouds. The 3.9 μm radiances for cloud-free
        land areas are averaged over 50 rows at a time to obtain an
        approximately latitudinal value. Wherever a cloud-covered pixel
        exceeds this value, it is flagged.

        Args:
            | ir039 (:obj:`ndarray`): Array for the 3.9 μm channel.
            | nir016 (:obj:`ndarray`): Array for the 1.6 μm channel.
            | vis006 (:obj:`ndarray`): Array for the 0.6 μm channel.
            | cloudmask (:obj:`MaskedArray`): Array for masked cloud objects.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Water Cloud Filter")
        # Weak water cloud test with NDSI
        self.ndsi_ci = (self.vis006 - self.nir016) / (self.vis006 +
                                                      self.nir016)
        water_mask = self.ndsi_ci > 0.1
        # Small droplet proxy test
        # Get only cloud free pixels
        cloud_free_ma = np.ma.masked_where(~self.cloudmask, self.ir039)
        # Latitudinal average cloud free radiances
        self.lat_cloudfree = np.ma.mean(cloud_free_ma, 1)
        logger.debug("Mean latitudinal threshold for cloudfree areas: %.2f K"
                     % np.mean(self.lat_cloudfree))
        self.line = 0
        # Apply latitudinal threshold to cloudy areas
        drop_mask = np.apply_along_axis(self.find_watercloud, 1, self.ir039,
                                        self.lat_cloudfree)

        # Create snow mask for image array
        self.mask = water_mask | drop_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def find_watercloud(self, lat, thres):
        """Funciton to compare row of BT with given latitudinal thresholds"""
        if not isinstance(lat, np.ma.masked_array):
            lat = np.ma.MaskedArray(lat, mask=np.zeros(lat.shape))
        if all(lat.mask):
            res = lat.mask
        elif np.ma.is_masked(thres[self.line]):
            res = lat <= np.mean(self.lat_cloudfree)
        else:
            res = lat <= thres[self.line]
        self.line += 1

        return(res)


class SpatialCloudTopHeightFilter_old(BaseArrayFilter):
    """Filtering cloud clusters by height for satellite images."""
    # Required inputs
    attrlist = ['ir108', 'clusters', 'cluster_z']

    def filter_function(self):
        """Cloud cluster filter routine

        This filter utilizes spatially clustered cloud objects and their
        cloud top height to mask cloud clusters with cloud top height above
        2000 m.
        """
        logger.info("Applying Spatial Clustering Cloud Top Height Filter")
        # Apply maximum threshold for cluster height to identify low fog clouds
        cluster_mask = self.clusters.mask
        for key, item in self.cluster_z.iteritems():
            if any([c > 2000 for c in item]):
                cluster_mask[self.clusters == key] = True

        # Create additional fog cluster map
        self.cluster_cth = np.ma.masked_where(cluster_mask, self.clusters)
        for key, item in self.cluster_z.iteritems():
            if all([c <= 2000 for c in item]):
                self.cluster_cth[self.cluster_cth == key] = np.mean(item)

        # Create cluster mask for image array
        self.mask = cluster_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class SpatialCloudTopHeightFilter(BaseArrayFilter):
    """Filtering cloud clusters by height for satellite images."""
    # Required inputs
    attrlist = ['cth', 'elev']

    def filter_function(self):
        """Cloud top height filter routine

        This filter uses given cloud top heights arrays for low clouds to mask
        cloud clusters with cloud top height above 1000 m in comparison to
        given ground elevation.

        Args:
            | cth (:obj:`ndarray`): Array of cloud top height in m.
            | elev (:obj:`ndarray`): Array of area elevation.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Spatial Cloud Top Height Filter")
        # Apply maximum threshold for cluster height to identify low fog clouds
        cth_mask = (self.cth - self.elev) > 1000

        # Create cluster mask for image array
        self.mask = cth_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class SpatialHomogeneityFilter(BaseArrayFilter):
    """Filtering cloud clusters by StDev for satellite images."""
    # Required inputs
    attrlist = ['ir108', 'clusters']

    def __init__(self, *args, **kwargs):
        super(SpatialHomogeneityFilter, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'maxsize'):
            self.maxsize = 10000  # Number of maximal clustered cloud pixel

    def filter_function(self):
        """Cloud cluster filter routine

        This filter utilizes spatially clustered cloud objects and their
        cloud top height to mask cloud clusters with  with spatial inhomogen
        cloud clusters. Cloud top temperature standard deviations less than 2.5
        are filtered.
        A maximal size is used to prevent filtering of large cloud clusters
        which exhibit bigger variability due to its size. The filter is only
        applied to cloud clusters below the given limit.

        Args:
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | clusters (:obj:`MaskedArray`): Masked array for cloud clusters.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Spatial Clustering Inhomogeneity Filter")
        # Surface homogeneity test
        cluster_mask = deepcopy(self.inmask)
        cluster, nlbl = ndimage.label(~self.clusters.mask)
        cluster_ma = np.ma.masked_where(self.inmask, self.clusters)

        cluster_sd = ndimage.standard_deviation(self.ir108, cluster_ma,
                                                index=np.arange(1, nlbl+1))

        # 4. Mask potential fog clouds with high spatial inhomogeneity
        sd_mask = cluster_sd > 2.5
        cluster_dict = {key: sd_mask[key - 1] for key in np.arange(1, nlbl+1)}
        for val in np.arange(1, nlbl+1):
            ncluster = np.count_nonzero(cluster_ma == val)
            if ncluster <= self.maxsize:
                cluster_mask[cluster_ma == val] = cluster_dict[val]
            else:
                logger.info("Exclude cloud cluster {} of size {} from filter"
                            .format(val, ncluster))
                cluster_mask[cluster_ma == val] = 0

        # Create cluster mask for image array
        self.mask = cluster_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class CloudPhysicsFilter(BaseArrayFilter):
    """Filtering cloud microphysics for satellite images."""
    # Required inputs
    attrlist = ['reff', 'cot']

    def filter_function(self):
        """Cloud microphysics filter routine

        Typical microphysical parameters for fog were taken from studies.
        Fog optical depth normally ranges between 0.15 and 30 while droplet
        effective radius varies between 3 and 12 μm, with a maximum of 20 μm in
        coastal fog. The respective maxima for optical depth (30) and droplet
        radius (20 μm) are applied to the low stratus mask as cut-off levels.
        Where a pixel previously identified as fog/low stratus falls outside
        the range it will now be flagged as a non-fog pixel.

        Args:
            | cot (:obj:`ndarray`): Array of cloud optical thickness (depth).
            | reff (:obj:`ndarray`): Array of cloud particle effective raduis.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Spatial Clustering Inhomogenity Filter")

        if np.ma.isMaskedArray(self.cot):
                self.cot = self.cot.base
        if np.ma.isMaskedArray(self.reff):
            self.reff = self.reff.base
        # Add mask by microphysical thresholds
        cpp_mask = (self.cot > 30) | (self.reff > 20e-6)
        # Create cloud physics mask for image array
        self.mask = cpp_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True


class LowCloudFilter(BaseArrayFilter):
    """Filtering low clouds for satellite images."""
    # Required inputs
    attrlist = ['lwp', 'cth', 'ir108', 'clusters', 'reff', 'elev']

    # Correction factor for 3.7 um LWP retrievals
    lwp_corr = 0.88  # Reference: (Platnick 2000)

    def __init__(self, *args, **kwargs):
        super(LowCloudFilter, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'single'):
            self.single = False
        if not hasattr(self, 'substitude'):
            self.substitude = True
        # Plot cbh and fbh results
        self.plotattr = ['cbh', 'fbh']

    def filter_function(self):
        """Cloud microphysics filter routine

        The filter separate low stratus clouds from ground fog clouds
        by computing the cloud base height with a 1D low cloud model for
        each cloud cluster.

        Args:
            | lwp (:obj:`ndarray`): Array of cloud liquid water path.
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | clusters (:obj:`MaskedArray`): Masked array for cloud clusters.
            | cth (:obj:`ndarray`): Array of cloud top height in m.
            | elev (:obj:`ndarray`): Array of area elevation.
            | reff (:obj:`ndarray`): Array of cloud particle effective raduis.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Low Cloud Filter")
        # Creating process pool
        pool = mp.Pool(self.nprocs)
        mlogger = mp.log_to_stderr()
        mlogger.setLevel(logging.DEBUG)
        # Declare result arrays without copy
        self.cbh = np.empty(self.clusters.shape, dtype=np.float)
        self.fbh = np.empty(self.clusters.shape, dtype=np.float)
        self.fog_mask = self.clusters.mask
        # Define mode of parallelisation
        if self.single:  # Run low cloud models parallized for single pixels
            count_cells = 0
            logger.info("Run low cloud models for single cells")
            applyres = []
            task_count = self.clusters.count()
            if self.plot:
                # Plot cloud top height distribution for clusters
                self.plot_cluster_stat()
            # Get pool result list
            self.result_list = []
            self.index_list = []
            # Loop over single cell processes
            for r, c in np.ndindex(self.clusters.squeeze().shape):
                if self.clusters.mask[r, c] == 0:
                    self.index_list.append((r, c))
                    count_cells += 1
                    workinput = [self.lwp[r, c], self.cth[r, c],
                                 self.ir108[r, c], self.reff[r, c]]
                    applyres.append(pool.apply_async(self.get_fog_base_height,
                                                     args=workinput,
                                                     callback=self.log_result))
            # Log tasks
            while True:
                incomplete_count = sum(1 for x in applyres if not x.ready())

                if incomplete_count == 0:
                    logger.info("All Done. Completed {} tasks"
                                .format(task_count))
                    break
                remain = float(task_count - incomplete_count) / task_count * 100
                logger.info("{} Tasks Remaining --- {:.2f} % Complete"
                            .format(incomplete_count, remain))
                time.sleep(1)
            # Wait for all processes to finish
            pool.close()
            pool.join()
            logger.info("Finished low cloud models for {} cells"
                        .format(count_cells))
            # Create ground fog and low stratus cloud masks and cbh
            for i, indices in enumerate(self.index_list):
                r, c = indices
                self.cbh[r, c] = self.result_list[i][0]
                self.fbh[r, c] = self.result_list[i][1]
            # Mask non ground fog clouds
            self.fog_mask[(self.fbh - self.elev > 0) \
                          | np.isnan(self.fbh)] = True

        else:  # Run low cloud models parallized aggregated for cloud clusters
            # Compute mean values for cloud clusters
            lwp_cluster = self.get_cluster_stat(self.clusters, self.lwp * 1000,
                                                exclude=[0])
            cth_cluster = self.get_cluster_stat(self.clusters, self.cth,
                                                exclude=[0])
            ctt_cluster = self.get_cluster_stat(self.clusters, self.ir108)
            reff_cluster = self.get_cluster_stat(self.clusters, self.reff, [],
                                                 False)
            if self.plot:
                # Plot cloud top height distribution for clusters
                self.plot_cluster_stat()
            # Loop over processes
            logger.info("Run low cloud models for cloud clusters")
            applyres = []
            task_count = len(lwp_cluster)
            # Get pool result list
            self.result_list = []
            for key in lwp_cluster.keys():
                workinput = [lwp_cluster[key], cth_cluster[key],
                             ctt_cluster[key], reff_cluster[key]]
                applyres.append(pool.apply_async(self.get_fog_base_height,
                                                 args=workinput,
                                                 callback=self.log_result))
            # Log tasks
            while True:
                incomplete_count = sum(1 for x in applyres if not x.ready())

                if incomplete_count == 0:
                    logger.info("All Done. Completed {} tasks"
                                .format(task_count))
                    break
                remain = float(task_count - incomplete_count) / task_count * 100
                logger.info("{} Tasks Remaining --- {:.2f} % Complete"
                            .format(incomplete_count, remain))
                time.sleep(1)
            # Wait for all processes to finish
            pool.close()
            pool.join()
            # Create ground fog and low stratus cloud masks and cbh
            keys = lwp_cluster.keys()
            for i, res in enumerate(self.result_list):
                self.cbh[self.clusters == keys[i]] = res[0]
                self.fbh[self.clusters == keys[i]] = res[1]
                # Mask non ground fog clouds
                self.fog_mask[(self.clusters == keys[i]) & (self.fbh -
                                                            self.elev >
                                                            0)] = True
        # Create cloud physics mask for image array
        self.mask = self.fog_mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def log_result(self, result):
        # This is called whenever a pool(i) returns a result.
        # Results are modified only by the main process, not the pool workers.
        self.result_list.append(result)

    def get_fog_base_height(self, cwp, cth, ctt, reff):
        """ Calculate fog base heights for low cloud pixels with a
        numerical 1-D low cloud model and known liquid water path, cloud top
        height / temperature and droplet effective radius from satellite
        retrievals.
        """
        lowcloud = LowWaterCloud(cth=cth,
                                 ctt=ctt,
                                 cwp=cwp * self.lwp_corr,
                                 cbh=0,
                                 reff=reff)
        try:
            # Calculate cloud base height
            cbh = lowcloud.get_cloud_base_height(-100, 'basin')
            # Get visibility and fog cloud base height
            fbh = lowcloud.get_fog_base_height(self.substitude)
        except Exception as e:
            logger.error(e, exc_info=True)
            cbh = np.nan
            fbh = np.nan

        return cbh, fbh

    def get_cluster_stat(self, clusters, values, exclude=[0],
                         noneg=True, stat='mean', data=False):
        """Calculate the mean of an array of values for given cluster
        structures.
        """
        # Optional statistic parameters
        stat_dict = {'mean': np.nanmean, 'std': np.nanstd, 'min': np.nanmin,
                     'max': np.nanmax, 'median': np.nanmedian}
        result = defaultdict(list)
        if np.ma.isMaskedArray(clusters):
            clusters = clusters.filled(0)
        # Calculate mean values for clusters
        for index, key in np.ndenumerate(clusters):
            if key != 0:

                val = values[index]
                if val in exclude:
                    # Remove exluced values
                    val = np.nan
                elif val < 0 and noneg:
                    # Optional remove of negative values
                    val = np.nan
                # Add value to result dict
                result[key].append(val)
        # Calculate average cluster values by dictionary key
        if not data:
            result = {k: stat_dict[stat](v) for k, v in result.iteritems()}

        return result

    def plot_cluster_stat(self, param=None, label='Cloud Top Height in m'):
        """Plot cloud top height distribution for cloud clusters"""
        if param is None:
            param = self.cth
        clusterdata = self.get_cluster_stat(self.clusters, param,
                                            exclude=[], noneg=False, data=True)
        data = [i for i in clusterdata.values() if len(i) >= 3]
        plt.figure(figsize=(14, 8))
        plt.tick_params(labelsize=8)
        plt.xticks(rotation=90)
        plt.boxplot(data)
        if self.save:
            savedir = os.path.join(self.dir, self.name + '_cluster_stat_' +
                                   datetime.strftime(self.time, '%Y%m%d%H%M') +
                                   '.png')
            plt.savefig(savedir)
            logger.info("Low cloud cluster distributions are plotted to: {}"
                        .format(savedir))
        else:
            plt.show()

        return(clusterdata)


class CloudMotionFilter(BaseArrayFilter):
    """Filtering FLS cloud by motion trakcing for satellite images."""
    # Required inputs
    attrlist = ['preir108', 'ir108']
    try:
        import cv2
    except:
        Warning("openCV Python package cv2 not found. Please install"
                "opencv and/or the cv-python interface")

    def filter_function(self):
        """Cloud motion filter routine

        Fog clouds are a stationary phenomena and therefore exhibit little
        observable cloud motion, Except in the stage of development and
        dissipation, where the motion should be different to the wind direction
        and in dependency og the terraim elevation.
        This filter utilizes atmospheric motion vectors that are calculated by
        an optical flow (tvl1 algorithm) approach to distinguish between
        stationary low clouds and moving non fog clouds.

        Args:
            | preir108 (:obj:`ndarray`): Array for the 10.8 μm channel for a
                                         preceding scene.
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Cloud Motion Filter")

        # Calculate motion vectors
        logger.info("Calculate optical flow...")

        optflow = self.cv2.createOptFlow_DualTVL1()
        # DWD Radar parameterization by M. Werner
        optflow.setEpsilon(0.01)
        optflow.setLambda(0.05)
        optflow.setOuterIterations(300)
        optflow.setInnerIterations(2)
        optflow.setGamma(0.1)
        optflow.setScalesNumber(5)
        optflow.setTau(0.25)
        optflow.setTheta(0.3)
        optflow.setWarpingsNumber(2)
        optflow.setScaleStep(0.5)
        optflow.setMedianFiltering(1)
        optflow.setUseInitialFlow(0)

        optflow.setWarpingsNumber(5)  # default
        optflow.setScaleStep(0.8)
        optflow.setLambda(0.15)

        # Rescale values to 0->1 CV_32F1
        min_val = min(np.ma.amin(self.ir108), np.ma.amin(self.preir108))
        max_val = max(np.ma.amax(self.ir108), np.ma.amax(self.preir108))

        ult = self.ir108 - min_val
        ult = ult / (max_val - min_val)
        ult = ult.astype(np.float32)

        penult = self.preir108 - min_val
        penult = penult / (max_val - min_val)
        penult = penult.astype(np.float32)

        flow = optflow.calc(penult.data, ult.data, None)
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]

        # Plot motion field
        img = self.draw_motion_vectors(flow)
        img.show()

        # Create cloud physics mask for image array
        self.mask = self.arr.mask

        self.result = np.ma.array(self.arr, mask=self.mask)

        return flow

    def draw_motion_vectors(self, flow, step=16):
        """Draw motion vectors generated by optical flow method."""
        img = self._plot_image('preir108')
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
        fx, fy = flow[y, x].T

        # create line endpoints
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)

        # Ceate image and draw
        out = deepcopy(img)
        # out = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in lines:
            self.cv2.arrowedLine(out, (x1, y1), (x2, y2), 255, 1)
            self.cv2.circle(out, (x1, y1), 1, 255, -1)

        return out


class StationFusionFilter(BaseArrayFilter):
    """Station data fusion filter for satellite images."""
    # Required inputs
    attrlist = ['ir108', 'ir039', 'lowcloudmask', 'cloudmask', 'elev',
                'bufrfile', 'time', 'area']

    def __init__(self, *args, **kwargs):
        super(StationFusionFilter, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'cloudmask'):
            self.cloudmask = self.get_cloudmask()
        if not hasattr(self, 'heightvar'):
            self.heightvar = 50  # in meters
        if not hasattr(self, 'fogthres'):
            self.fogtrhes = 1000  # in meters
        if not hasattr(self, 'limit'):
            self.limit = False
        # Remove nodata values from elevation
        self.elev[self.elev == 9999.0] = np.nan
        # Plot additional filter results
        self.plotattr = ['lowcloudmask', 'cloudmask', 'fogmask', 'nofogmask',
                         'missdemmask', 'falsedemmask']

    def filter_function(self):
        """Station data fusion filter routine

        This filter provide a data fusion approach for satellite derived low
        cloud masked raster data with vector data of visibilities observed by
        sensors at weather stations. This raster/vector data fusion is done in
        several steps.
        First a BUFR file with WMO standardized weather station data is
        imported. Next the visibility measurements are extracted and resampled
        to the given cloud mask raster array shape. Then a fog mask for the
        station data is created and a DEM based interpolation is performed.

        Afterwards the stations are compared to the low cloud and cloud masks
        and case dependant mask corrections are performed.

        Args:
            | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
            | ir039 (:obj:`ndarray`): Array for the 3.9 μm channel.
            | lowcloudmask (:obj:`ndarray`): Low cloud mask array.
            | cloudmask (:obj:`ndarray`): General cloud mask array.
            | elev (:obj:`ndarray`): Array of elevation.
            | bufrfile (:obj:`str`): Path to weather station BUFR-file.
            | time (:obj:`Datetime`): Time instance of station data
                                      as *datetime* object.
            | area (:obj:`area_def`): Corresponding area definition for
                                      cloud masks.
            | heightvar (:obj:`float`): Height variance for fog masking in m.
                                        Default is 50 m.
            | fogthres (:obj:`float`): Visibility threshold for fog masking
                                       in m. Default is 1000 m.
            | limit (:obj:`bool`): Boolean to limit the output region to
                                   station data coverage.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Station Data Fusion Filter")

        # 1. Import BUFR file
        stations = read_synop(self.bufrfile, 'visibility')
        currentstations = stations[self.time.strftime("%Y%m%d%H0000")]
        lats = [i[2] for i in currentstations]
        lons = [i[3] for i in currentstations]
        vis = [i[4] for i in currentstations]

        # 2. Port visibility vector data to raster array shape
        self.visarr = np.empty(self.arr.shape[:2])
        self.visarr.fill(np.nan)

        x, y = (self.area.get_xy_from_lonlat(lons, lats))
        xmin, xmax, ymin, ymax = [np.nanmin(x), np.nanmax(x), np.nanmin(y),
                                  np.nanmax(y)]
        vis_ma = np.ma.array(vis, mask=x.mask)
        self.visarr[y.compressed(), x.compressed()] = vis_ma.compressed()

        # Mask out fog cells
        fogstations = self.visarr <= 1000
        nofogstations = self.visarr > 1000

        # 3. Compare with cloud clusters
        cloudcluster, ncloudclst = ndimage.label(~self.cloudmask)
        lowcluster, nlowclst = ndimage.label(~self.lowcloudmask)
        cloudtrue = cloudcluster[fogstations]
        lowcloudfog = lowcluster[fogstations]
        lowcloudnofog = lowcluster[nofogstations]

        # Compare stations with low cloud mask
        firstvalid = self.validate_fogmask(fogstations, nofogstations,
                                           lowcluster, nlowclst, False,
                                           elev=self.elev)
        lchit, lcmiss, lcfalse, lctrue = firstvalid
        # Negate validation components for optional attribute plotting
        self.lchit = ~lchit
        self.lcmiss = ~lcmiss
        self.lcfalse = ~lcfalse
        self.lctrue = ~lctrue

        # 4. Interpolate station fog mask based on DEM
        self.missdemmask, missvalid = self.interpolate_dem(lcmiss,
                                                           self.elev,
                                                           self.heightvar,
                                                           self.cloudmask)
        self.falsedemmask, falsevalid = self.interpolate_dem(lcfalse,
                                                             self.elev,
                                                             self.heightvar,
                                                             self.cloudmask)
        # 5. Perfom data fusion with low cloud mask
        self.mask = np.copy(self.lowcloudmask)
        # Station false alarm cases  --> Remove DEM based
        self.mask[~self.falsedemmask & (falsevalid >= missvalid)] = True
        # Station Miss cases --> Add DEM based mask
        self.mask[~self.missdemmask & (missvalid > falsevalid)] = False

        # 6. Create fog cloud mask for image array
        if self.limit:
            logger.info("Limit mask to station data region x: {} - {},"
                        " y: {} - {}".format(xmin, xmax, ymin, ymax))
            # Crop to limited region with station coverage
            l = (ymin, ymax, xmin, xmax)

            self.arr = self.arr[l[0]:l[1], l[2]:l[3]]
            self.inmask = self.inmask[l[0]:l[1], l[2]:l[3]]
            self.mask = self.mask[l[0]:l[1], l[2]:l[3]]
            self.missdemmask = self.missdemmask[l[0]:l[1], l[2]:l[3]]
            self.falsedemmask = self.falsedemmask[l[0]:l[1], l[2]:l[3]]
            self.cloudmask = self.cloudmask[l[0]:l[1], l[2]:l[3]]
            self.lowcloudmask = self.lowcloudmask[l[0]:l[1], l[2]:l[3]]
            fogstations = fogstations[l[0]:l[1], l[2]:l[3]]
            nofogstations = nofogstations[l[0]:l[1], l[2]:l[3]]

        self.fogmask = ~fogstations
        self.nofogmask = ~nofogstations

        lowcluster, nlowclst = ndimage.label(~self.mask)
        secondvalid = self.validate_fogmask(fogstations, nofogstations,
                                            lowcluster.squeeze(),
                                            nlowclst, True, firstvalid,
                                            elev=self.elev)
        # Return filtered output with mask
        self.result = np.ma.array(self.arr, mask=self.mask)

        return True

    def interpolate_dem(self, stations, elev, heightvar, mask=None):
        """Interpolate a fog mask for stations based on DEM information.

        Args:
            | stations (:obj:`ndarray`): Boolean array of foggy stations.
            | elev (:obj:`ndarray`): Array for elevation information (DEM).
            | heightvar (:obj:`ndarray`): Height variance for fog masking in m.
            | mask (:obj:`ndarray`): Boolean array with cloud mask,
                                          optional.

        Returns:
            Interpolated fog mask
        """
        # Get elevation for foggy stations
        fogelev = elev[stations]
        fogrow, fogcol = np.where(stations)
        # Init fog station DEM mask
        demmask = np.ones(elev.shape[:2], dtype=bool)
        # Setup validation mask to track number of station per masked pixel
        validmask = np.zeros(elev.shape[:2], dtype=int)
        # Loop over foggy stations and extract heightbased fog mask
        for i in np.arange(len(fogrow)):
            # Only values within the given variance around the station
            # elevation are masked.
            elevmask = np.logical_and(elev >= fogelev[i] - heightvar,
                                      elev <= fogelev[i] + heightvar)
            elevcluster, nfogclst = ndimage.label(elevmask)
            clstnum = elevcluster[fogrow[i], fogcol[i]]
            selectclst = elevcluster == clstnum
            demmask[selectclst.squeeze()] = False
            # Cumulate validation mask
            validmask[selectclst.squeeze()] += 1
        logger.info("Interpolated DEM based fog mask for {} stations"
                    .format(len(fogrow)))

        # Filter by cloud mask
        if mask is not None:
            demmask[~mask.squeeze()] = True
            validmask[demmask] = 0

        return(demmask, validmask)

    @classmethod
    def validate_fogmask(self, fogstations, nofogstations, lowcluster,
                         nlowclst, plot=True, compare=[], elev=None):
        """Using station data to validate a low cloud/ fog mask.

        Args:
            | fogstations (:obj:`ndarray`): Boolean array of foggy stations.
            | nofogstations (:obj:`ndarray`): Boolean array of no-fog stations.
            | lowcluster (:obj:`ndarray`): Array of enumerated low cloud
                                           clusters.
            | nlowclst (:obj:`int`): Number of low cloud clusters.
            | plot (:obj:`bool`): Boolean if validation results is plotted as
                                  logger message.
            | compare (:obj:`list`): Optional ordered validation result list:
                                     hits, misses, false-alarm, true-negative
                                     as boolean numpy ndarrays.
            | elev (:obj:`ndarray`): Array of elevation information

        Returns:
            List of validation results
        """
        if elev is None:
            elev = self.elev
        lowcloudhit = np.logical_and(fogstations, lowcluster != 0)
        lowcloudmiss = np.logical_and(fogstations, lowcluster == 0)
        lowcloudfalse = np.logical_and(nofogstations, lowcluster != 0)
        lowcloudtrueneg = np.logical_and(nofogstations, lowcluster == 0)

        if plot:
            stationsum = np.nansum(fogstations) + np.nansum(nofogstations)
            allelev = elev[np.logical_or(fogstations, nofogstations)]
            fogelev = elev[fogstations]
            nofogelev = elev[nofogstations]
            logmsg = "\n         ------ Station validation ------\n" \
                     "    Station statistics   | Elevation in m\n" \
                     "                       n | min    mean    max\n" \
                     "    Stations:     {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "        Fog:      {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "        No Fog:   {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    Low clouds:   {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    --------------------------------------------\n" \
                     "    Hits:         {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    Misses:       {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    False alarm:  {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    True negativ: {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                     "    --------------------------------------------" \
                     .format(stationsum, np.nanmin(self.check_zerolist(allelev)),
                             np.nanmean(self.check_zerolist(allelev)),
                             np.nanmax(self.check_zerolist(allelev)),
                             np.nansum(self.check_zerolist(fogstations)),
                             np.nanmin(self.check_zerolist(fogelev)),
                             np.nanmean(self.check_zerolist(fogelev)),
                             np.nanmax(self.check_zerolist(fogelev)),
                             np.nansum(self.check_zerolist(nofogstations)),
                             np.nanmin(self.check_zerolist(nofogelev)),
                             np.nanmean(self.check_zerolist(nofogelev)),
                             np.nanmax(self.check_zerolist(nofogelev)),
                             nlowclst, np.min(self.check_zerolist(elev[lowcluster != 0])),
                             np.nanmean(self.check_zerolist(elev[lowcluster != 0])),
                             np.nanmax(self.check_zerolist(elev[lowcluster != 0])),
                             np.nansum(self.check_zerolist(lowcloudhit)),
                             np.nanmin(self.check_zerolist(elev[lowcloudhit])),
                             np.nanmean(self.check_zerolist(elev[lowcloudhit])),
                             np.nanmax(self.check_zerolist(elev[lowcloudhit])),
                             np.nansum(self.check_zerolist(lowcloudmiss)),
                             np.nanmin(self.check_zerolist(elev[lowcloudmiss])),
                             np.nanmean(self.check_zerolist(elev[lowcloudmiss])),
                             np.nanmax(self.check_zerolist(elev[lowcloudmiss])),
                             np.nansum(self.check_zerolist(lowcloudfalse)),
                             np.nanmin(self.check_zerolist(elev[lowcloudfalse])),
                             np.nanmean(self.check_zerolist(elev[lowcloudfalse])),
                             np.nanmax(self.check_zerolist(elev[lowcloudfalse])),
                             np.nansum(self.check_zerolist(lowcloudtrueneg)),
                             np.nanmin(self.check_zerolist(elev[lowcloudtrueneg])),
                             np.nanmean(self.check_zerolist(elev[lowcloudtrueneg])),
                             np.nanmax(self.check_zerolist(elev[lowcloudtrueneg])))

            if compare:
                compmsg = "\n         ------ Comparision validation ------\n" \
                          "    Hits:         {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                          "    Misses:       {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                          "    False alarm:  {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                          "    True negativ: {:6d} | {:6.2f} {:6.2f} {:6.2f}\n" \
                          "    --------------------------------------------" \
                          .format(np.nansum(self.check_zerolist(compare[0])),
                                  np.nanmin(self.check_zerolist(elev[compare[0]])),
                                  np.nanmean(self.check_zerolist(elev[compare[0]])),
                                  np.nanmax(self.check_zerolist(elev[compare[0]])),
                                  np.nansum(self.check_zerolist(compare[1])),
                                  np.nanmin(self.check_zerolist(elev[compare[1]])),
                                  np.nanmean(self.check_zerolist(elev[compare[1]])),
                                  np.nanmax(self.check_zerolist(elev[compare[1]])),
                                  np.nansum(self.check_zerolist(compare[2])),
                                  np.nanmin(self.check_zerolist(elev[compare[2]])),
                                  np.nanmean(self.check_zerolist(elev[compare[2]])),
                                  np.nanmax(self.check_zerolist(elev[compare[2]])),
                                  np.nansum(self.check_zerolist(compare[3])),
                                  np.nanmin(self.check_zerolist(elev[compare[3]])),
                                  np.nanmean(self.check_zerolist(elev[compare[3]])),
                                  np.nanmax(self.check_zerolist(elev[compare[3]])))
                logger.info(logmsg + compmsg)
            else:
                logger.info(logmsg)

        return([lowcloudhit, lowcloudmiss, lowcloudfalse, lowcloudtrueneg])

    def get_cloudmask(self):
        """Get cloud filter mask."""
        # Create cloud mask
        cloudfilter = CloudFilter(self.ir108, ir108=self.ir108,
                                  ir039=self.ir039)
        cloudimg, cloudmask = cloudfilter.apply()

        return(cloudmask)

    @classmethod
    def check_zerolist(self, inlist):
        if len(inlist) == 0:
            return([np.nan])
        else:
            return(inlist)


class NumericalModelFilter(BaseArrayFilter):
    """NWP filtering for satellite images."""
    # Required inputs
    attrlist = ['t_model', 'td_model']

    def __init__(self, *args, **kwargs):
        super(NumericalModelFilter, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'tdthres'):
            self.tdthres = 2.2

    def filter_function(self):
        """Numerical model filter routine

        The 2 meter temperature and dew point data from a numerical weather
        model can be used to exclude regions with dew point differences
        that prevent fog development.
        This filter utilize the modelled temperature data to filter cells with:

            - Dew point differences higher than 2.2 K
              Source: http://glossary.ametsoc.org/wiki/Fog

        Args:
            | t_model (:obj:`ndarray`): Array for the 2 meter temperature.
            | td_model (:obj:`ndarray`): Array for the dew point temperature.

        Returns:
            Filter image and filter mask.
        """
        logger.info("Applying Numerical Model Filter")
        # Calculate dew point differences
        self.tdiff = self.t_model - self.td_model

        # Create snow mask for image array
        tdiff_thres = (self.tdiff >= self.tdthres)

        # Create snow mask for image array
        self.mask = tdiff_thres

        self.result = np.ma.array(self.arr, mask=self.mask)

        return True
