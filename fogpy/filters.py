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

import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks_cwt

logger = logging.getLogger(__name__)


class NotApplicableError(Exception):
    """Exception to be raised when a filter is not applicable."""
    pass


class BaseArrayFilter(object):
    """This super filter class provide all functionalities to apply a filter
    funciton on a given numpy array representing a satellite image and return
    the filtered masked array as result"""
    def __init__(self, arr, **kwargs):
        if isinstance(arr, np.ma.MaskedArray):
            self.arr = arr
            self.inmask = arr.mask
        elif isinstance(arr, np.ndarray):
            self.arr = arr
        else:
            raise ImportError('The filter <{}> needs a valid 2d numpy array '
                              'as input'.format(self.__class__.__name__))
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                self.__setattr__(key, value)
            self.result = None
            self.mask = None

    def apply(self):
        """Apply the given filter function"""
        if self.isapplicable():
            self.filter_function()
            self.check_results()
        else:
            raise NotApplicableError('Array filter <{}> is not applicable'
                                     .format(self.__class__.__name__))

        return(self.result, self.mask)

    def isapplicable(self):
        """Test filter applicability"""
        ret = True
        return(ret)

    def filter_function(self):
        """Filter routine"""
        self.mask = np.ones(self.arr.shape) == 1

        self.result = np.ma.array(self.arr, mask=self.mask)

        return(True)

    def check_results(self):
        """Check filter results for plausible results"""
        ret = True
        return(ret)


class CloudFilter(BaseArrayFilter):
    """Cloud filtering for satellite images.
    """
    def isapplicable(self):
        """Test filter applicability"""
        attrlist = ['ir108', 'ir039']
        ret = []
        for attr in attrlist:
            if hasattr(self, attr):
                ret.append(True)
            else:
                ret.append(False)

        return(all(ret))

    def filter_function(self):
        """Cloud Filter routine

        Given the combination of a solar and a thermal signal at 3.9 μm,
        the difference in radiances to the 10.8 μm must be larger for a
        cloud-contaminated pixel than for a clear pixel.
        In the histogram of the difference the clear sky peak is identified
        within a certain range. The nearest significant relative minimum in the
        histogram towards more negative values is detected and used as a
        threshold to separate clear from cloudy pixels in the image.
        """
        logger.info("### Applying fog cloud filters to input arrays ###")

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
        peakrange = self.hist[1][peakind][(self.hist[1][peakind] >= -10) &
                                          (self.hist[1][peakind] < 10)]
        self.minpeak = np.min(peakrange)
        self.maxpeak = np.max(peakrange)

        # Determine threshold
        logger.debug("Histogram range for cloudy/clear sky pixels: {} - {}"
                     .format(self.minpeak, self.maxpeak))
        thres_index = localmin[(self.hist[1][localmin] <= self.maxpeak) &
                               (self.hist[1][localmin] >= self.minpeak) &
                               (self.hist[1][localmin] < 0.5)]
        self.thres = np.max(self.hist[1][thres_index])

        if self.thres > 0 or self.thres < -5:
            logger.warning("Cloud maks difference threshold {} outside normal"
                           " range (from -5 to 0)".format(self.thres))
        else:
            logger.debug("Cloud mask difference threshold set to %s"
                         .format(self.thres))
        # Create cloud mask for image array
        self.mask = self.cm_diff > self.thres

        self.result = np.ma.array(self.arr, mask=self.mask)

        return(True)

    def check_results(self):
        """Check filter results for plausible results"""
        ret = True
        return(ret)

    def plot_cloud_hist(self):
        plt.bar(self.hist[1][:-1], self.hist[0])
        plt.title("Histogram with 'auto' bins")
        plt.show()
