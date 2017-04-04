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

"""This module implements an base satellite algorithm class"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import get_cmap
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import measurements

from filters import CloudFilter
from filters import SnowFilter
from filters import IceCloudFilter
from filters import CirrusCloudFilter
from filters import WaterCloudFilter
from filters import SpatialCloudTopHeightFilter
from filters import SpatialHomogeneityFilter
from filters import CloudPhysicsFilter
from filters import LowCloudFilter

logger = logging.getLogger(__name__)


class NotProcessibleError(Exception):
    """Exception to be raised when a filter is not applicable."""
    pass


class BaseSatelliteAlgorithm(object):
    """This super filter class provide all functionalities to run an algorithm
    on satellite image arrays and return a new array as result"""
    def __init__(self, **kwargs):
        self.mask = None
        self.result = None
        self.attributes = []
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                self.attributes.append(key)
                if isinstance(value, np.ma.MaskedArray):
                    self.add_mask(value.mask)
                    value = self.check_dimension(value)
                    self.shape = value.shape
                elif isinstance(value, np.ndarray):
                    value = self.check_dimension(value)
                    self.shape = value.shape

                self.__setattr__(key, value)

    def run(self):
        """Start the algorithm and return results"""
        if self.isprocessible():
            self.procedure()
            self.check_results()
        else:
            raise NotProcessibleError('Satellite algorithm <{}> is not '
                                      'processible'
                                      .format(self.__class__.__name__))

        return self.result, self.mask

    def isprocessible(self):
        """Test runability here"""
        ret = True

        return(ret)

    def procedure(self):
        """Define algorithm procedure here"""
        self.mask = np.ones(self.shape) == 1

        self.result = np.ma.array(np.ones(self.shape), mask=self.mask)

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        ret = True
        return ret

    def add_mask(self, mask):
        """Compute the new array mask as union of all input array masks
        and computed masks"""
        if not np.ma.is_mask(mask):
            raise ImportError("Mask type is invalid")
        if self.mask is not None:
            self.mask = self.mask | mask
        else:
            self.mask = mask

    def get_kwargs(self, keys):
        """Return dictionary with passed keyword arguments"""
        return({key: self.__getattribute__(key) for key in self.attributes
                if key in keys})

    def plot_result(self):
        """Plotting the filter result"""
        cmap = get_cmap('gray')
        cmap.set_bad('goldenrod', 1.)
        imgplot = plt.imshow(self.result.squeeze(), cmap=cmap)
        plt.show()

    def check_dimension(self, arr):
        """ Check and convert arrays to 2D """
        if arr.ndim != 2:
            try:
                result = arr.squeeze()  # Try to reduce dimension
            except:
                raise ValueError("need 2-D input")
        else:
            result = arr

        return(result)


class FogLowStratusAlgorithm(BaseSatelliteAlgorithm):
    """This algorithm implements a fog and low stratus detection and forecasting
     for geostationary satellite images from the SEVIRI instrument onboard of
     METEOSAT second generation MSG satellites.
     It is utilizing the methods proposed in different innovative studies:

         - A novel approach to fog/low stratus detection using Meteosat 8 data
            J. Cermak & J. Bendix
        - Detecting ground fog from space – a microphysics-based approach
            J. Cermak & J. Bendix

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
    provided satellite images. Afterwards a separation between fog and low
    clouds are made by calibrating a cloud base height with a low cloud model
    to satellite retrieval information. Then a fog dissipation and subsequently
    a nowcasting of fog can be done.

            Input: Calibrated satellite images >-----   Implemented:
                                                    |
                1.  Cloud masking -------------------    yes
                                                    |
                2.  Snow masking --------------------    yes
                                                    |
                3.  Ice cloud masking ---------------    yes
                                                    |
                4.  Thin cirrus masking -------------    yes
                                                    |
                5.  Watercloud test -----------------    yes
                                                    |
                6.  Spatial clustering---------------    yes
                                                    |
                7.  Maximum margin elevation --------    yes
                                                    |
                8.  Surface homogenity check --------    yes
                                                    |
                9.  Microphysics plausibility check -    yes
                                                    |
                10.  Differenciate fog - low status -    ...
                                                    |
                11.  Fog dissipation ----------------
                                                    |
                12.  Nowcasting ---------------------
                                                    |
            Output: fog and low stratus mask <-------
     """
    def isprocessible(self):
        """Test runability here"""
        attrlist = ['ir108', 'ir039', 'vis008', 'nir016', 'vis006', 'ir087',
                    'ir120', 'lat', 'lon', 'time', 'elev', 'lwp']
        ret = []
        for attr in attrlist:
            if hasattr(self, attr):
                ret.append(True)
            else:
                ret.append(False)
                logger.warning("Missing input attribute: {}".format(attr))

        return all(ret)

    def procedure(self):
        """ Apply different filters and low cloud model to input data"""
        logger.info("Starting fog and low cloud detection algorithm")
        # 1. Cloud filtering
        cloud_input = self.get_kwargs(['ir108', 'ir039'])
        cloudfilter = CloudFilter(cloud_input['ir108'], **cloud_input)
        cloudfilter.apply()
        self.add_mask(cloudfilter.mask)

        # 2. Snow filtering
        snow_input = self.get_kwargs(['ir108', 'vis008', 'nir016', 'vis006'])
        snowfilter = SnowFilter(cloudfilter.result, **snow_input)
        snowfilter.apply()
        self.add_mask(snowfilter.mask)

        # 3. Ice cloud detection
        # Ice cloud exclusion - Only warm fog (i.e. clouds in the water phase)
        # are considered. Warning: No ice fog detection wiht this filter option
        ice_input = self.get_kwargs(['ir120', 'ir087', 'ir108'])
        icefilter = IceCloudFilter(snowfilter.result, **ice_input)
        icefilter.apply()
        self.add_mask(icefilter.mask)

        # 4. Cirrus cloud filtering
        cirrus_input = self.get_kwargs(['ir120', 'ir087', 'ir108', 'lat',
                                        'lon', 'time'])
        cirrusfilter = CirrusCloudFilter(icefilter.result, **cirrus_input)
        cirrusfilter.apply()
        self.add_mask(cirrusfilter.mask)

        # 5. Water cloud filtering
        water_input = self.get_kwargs(['ir108', 'vis008', 'nir016', 'vis006',
                                       'ir039'])
        waterfilter = WaterCloudFilter(icefilter.result,
                                       cloudmask=cloudfilter.mask,
                                       **water_input)
        waterfilter.apply()
        self.add_mask(waterfilter.mask)

        # 6. Spatial clustering
        clusters = self.get_cloud_cluster(self.mask)

        # 7. Calculate cloud top height
        bt_clear = np.ma.masked_where((~cloudfilter.mask |
                                       snowfilter.mask),
                                      self.ir108)
        bt_cloud = np.ma.masked_where(self.mask, self.ir108)
        self.cluster_z = self.get_lowcloud_cth(clusters, bt_clear, bt_cloud,
                                               self.elev)
        # Apply cloud top height filter
        cthfilter = SpatialCloudTopHeightFilter(waterfilter.result,
                                                ir108=self.ir108,
                                                clusters=clusters,
                                                cluster_z=self.cluster_z)
        cthfilter.apply()
        self.cluster_cth = cthfilter.cluster_cth
        self.add_mask(cthfilter.mask)

        # 8. Test spatial inhomogeneity
        stdevfilter = SpatialHomogeneityFilter(cthfilter.result,
                                               ir108=self.ir108,
                                               clusters=clusters)
        stdevfilter.apply()
        self.add_mask(stdevfilter.mask)

        # 9. Apply cloud microphysical filter
        physic_input = self.get_kwargs(['cot', 'reff'])
        physicfilter = CloudPhysicsFilter(stdevfilter.result,
                                          **physic_input)
        physicfilter.apply()
        self.add_mask(physicfilter.mask)

        # 10. Fog - low stratus cloud differentiation
        # Recalculate clusters
        clusters = self.get_cloud_cluster(self.mask)
        lowcloudfilter = LowCloudFilter(physicfilter.result,
                                        lwp=self.lwp,
                                        cth=self.cluster_cth,
                                        ir108=self.ir108,
                                        clusters=clusters)
        lowcloudfilter.apply()

        # Set results
        self.result = lowcloudfilter.cbh
        self.mask = physicfilter.mask

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        ret = True
        return ret

    def get_cloud_cluster(self, mask):
        """ Enumerate low water cloud clusters by spatial vicinity

        A mask is provided and the non masked values are spatially clustered
        using scipy label method

        Returns: Array with enumerated clusters
        """
        logger.info("Clustering low clouds")
        # Enumerate fog cloud clusters
        cluster = measurements.label(~mask)
        # Get 10.8 channel sampled by the previous fog filters
        result = np.ma.masked_where(mask, cluster[0])
        # Check dimension
        if result.ndim != 2:
            try:
                result = result.squeeze()  # Try to reduce dimension
            except:
                raise ValueError("need 2-D input")
        logger.debug("Number of spatial coherent fog cloud clusters: %s"
                     % np.nanmax(np.unique(result)))

        return result

    def sliding_window(self, arr, window_size):
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            try:
                arr = arr.squeeze()  # Try to reduce dimension
            except:
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

    def cell_neighbors(self, arr, i, j, d, value):
        """Return d-th neighbors of cell (i, j)"""
        if arr.ndim != 2:
            try:
                arr = arr.squeeze()  # Try to reduce dimension
            except:
                raise ValueError("need 2-D input")
        w = self.sliding_window(arr, 2*d+1)

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

        return center, neighbors

    def get_lowcloud_cth(self, cluster, cf_arr, bt_cc, elevation):
        """Get neighboring cloud free BT and elevation values of potenital
        fog cloud clusters and compute cloud top height from maximum BT
        differences for fog cloud contaminated pixel in comparison to cloud
        free areas and their corresponding elevation using a constant
        atmospheric lapse rate
        """
        from collections import defaultdict
        result = defaultdict(list)

        # Convert masked values to nan and zeros for clusters
        if np.ma.isMaskedArray(cf_arr):
            cf_arr = cf_arr.filled(np.nan)
        if np.ma.isMaskedArray(cluster):
            cluster = cluster.filled(0)

        for index, val in np.ndenumerate(cluster):
            if val != 0:
                # Get list of cloud free neighbor pixel
                tcc, tneigh = self.cell_neighbors(cf_arr, i=index[0],
                                                  j=index[1], d=1,
                                                  value=bt_cc)
                zcc, zneigh = self.cell_neighbors(elevation, i=index[0],
                                                  j=index[1], d=1,
                                                  value=elevation)
                tcf_diff = np.array([tcf - tcc for tcf in tneigh])
                zcf_diff = np.array([zcf - zcc for zcf in zneigh])
                # Get maximum bt difference
                try:
                    maxd = np.nanargmax(tcf_diff)
                except ValueError:
                    continue
                # Compute cloud top height with constant atmosphere temperature
                # lapse rate
                rate = 0.65
                cth = tcf_diff[maxd] / rate * 100 - zcf_diff[maxd]
                result[val].append(cth)

        return result
