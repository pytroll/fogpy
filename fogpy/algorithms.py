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
import os

from copy import deepcopy
from datetime import datetime
from matplotlib.cm import get_cmap
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import measurements
from scipy import interpolate
from filters import CloudFilter
from filters import SnowFilter
from filters import IceCloudFilter
from filters import CirrusCloudFilter
from filters import WaterCloudFilter
from filters import SpatialCloudTopHeightFilter
from filters import SpatialHomogeneityFilter
from filters import CloudPhysicsFilter
from filters import LowCloudFilter
from pyresample import image, geometry

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
        # Get class name
        self.name = self.__str__().split(' ')[0].split('.')[-1]
        # Set plotting attribute
        if not hasattr(self, 'save'):
            self.save = False
        if not hasattr(self, 'plot'):
            self.plot = False
        if not hasattr(self, 'dir'):
            self.dir = '/tmp'
        if not hasattr(self, 'resize'):
            self.resize = 1
        if not hasattr(self, 'plotrange'):
            self.plotrange = (0, 1)

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
        if self.plot:
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize)
        return True

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

    def plot_result(self, array=None, save=False, dir="/tmp", resize=1):
        """Plotting the algorithm result"""
        # Using Trollimage if available, else matplotlib is used to plot
        try:
            from trollimage.image import Image
            from trollimage.colormap import rainbow
            from trollimage.colormap import ylorrd
            from trollimage.colormap import Colormap
        except:
            logger.info("{} results can't be plotted to: {}". format(self.name,
                                                                     dir))
            return 0
        # Create image from data
        if array is None:
            if np.nanmax(self.result) > 1:
                self.plotrange = (np.nanmin(self.result),
                                  np.nanmax(self.result))
            result_img = Image(self.result.squeeze(), mode='L',
                               fill_value=None)
        else:
            self.plotrange = (np.nanmin(array), np.nanmax(array))
            result_img = Image(array.squeeze(), mode='L', fill_value=None)
        result_img.stretch("crude")
        # Colorize image
        # Define custom fog colormap
        customcol = Colormap((0., (250 / 255.0, 200 / 255.0, 40 / 255.0)),
                             (1., (1.0, 1.0, 229 / 255.0)),
                             (self.plotrange[1], (0.0, 1.0, 229 / 255.0)))
        ylorrd.set_range(*self.plotrange)
        logger.info("Set color range to {}".format(self.plotrange))
#         result_img.colorize(ylorrd)
        result_img.resize((self.result.shape[0] * int(resize),
                           self.result.shape[1] * int(resize)))
        if save:
            # Get output directory and image name
            savedir = os.path.join(dir, self.name + '_' +
                                   datetime.strftime(self.time,
                                                     '%Y%m%d%H%M') + '.png')
            result_img.save(savedir)
            logger.info("{} results are plotted to: {}". format(self.name,
                                                                self.dir))
        else:
            result_img.show()

        return(result_img)

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

    def plot_clusters(self, save=False, dir="/tmp"):
        """Plot the cloud clusters"""
        # Get output directory and image name
        name = self.__class__.__name__
        savedir = os.path.join(dir, name + '_clusters_' +
                               datetime.strftime(self.time,
                                                 '%Y%m%d%H%M') + '.png')
        # Using Trollimage if available, else matplotlib is used to plot
        try:
            from trollimage.image import Image
            from trollimage.colormap import rainbow
        except:
            logger.info("{} results can't be plotted to: {}". format(name,
                                                                     self.dir))
            return 0
        # Create image from data
        cluster_img = Image(self.clusters.squeeze(), mode='L', fill_value=None)
        cluster_img.stretch("crude")
        cluster_img.colorize(rainbow)
        cluster_img.resize((self.clusters.shape[0] * 5,
                            self.clusters.shape[1] * 5))
        if save:
            cluster_img.save(savedir)
            logger.info("{} results are plotted to: {}". format(name,
                                                                self.dir))
        else:
            cluster_img.show()


class DayFogLowStratusAlgorithm(BaseSatelliteAlgorithm):
    """This algorithm implements a fog and low stratus detection and forecasting
     for geostationary satellite images from the SEVIRI instrument onboard of
     METEOSAT second generation MSG satellites. Seven MSG channels from the
     solar and infrared spectra are used. Therefore the algorithm is applicable
     for daytime scenes only.
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
                10.  Differenciate fog - low status -    yes
                                                    |
                11.  Fog dissipation ----------------
                                                    |
                12.  Nowcasting ---------------------
                                                    |
            Output: fog and low stratus mask <-------
     """
    def __init__(self, *args, **kwargs):
        super(DayFogLowStratusAlgorithm, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'single'):
            self.single = False

    def isprocessible(self):
        """Test runability here"""
        attrlist = ['ir108', 'ir039', 'vis008', 'nir016', 'vis006', 'ir087',
                    'ir120', 'lat', 'lon', 'time', 'elev', 'lwp', 'reff']
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
        cloud_input = self.get_kwargs(['ir108', 'ir039', 'time', 'save',
                                       'resize', 'plot', 'dir'])
        cloudfilter = CloudFilter(cloud_input['ir108'], bg_img=self.ir108,
                                  **cloud_input)
        cloudfilter.apply()
        self.add_mask(cloudfilter.mask)

        # 2. Snow filtering
        snow_input = self.get_kwargs(['ir108', 'vis008', 'nir016', 'vis006',
                                      'time', 'save', 'resize', 'plot', 'dir'])
        snowfilter = SnowFilter(cloudfilter.result, bg_img=self.ir108,
                                **snow_input)
        snowfilter.apply()
        self.add_mask(snowfilter.mask)

        # 3. Ice cloud detection
        # Ice cloud exclusion - Only warm fog (i.e. clouds in the water phase)
        # are considered. Warning: No ice fog detection with this filter option
        ice_input = self.get_kwargs(['ir120', 'ir087', 'ir108', 'time', 'save',
                                     'resize', 'plot', 'dir'])
        icefilter = IceCloudFilter(snowfilter.result, bg_img=self.ir108,
                                   **ice_input)
        icefilter.apply()
        self.add_mask(icefilter.mask)

        # 4. Cirrus cloud filtering
        cirrus_input = self.get_kwargs(['ir120', 'ir087', 'ir108', 'lat',
                                        'lon', 'time', 'save', 'resize',
                                        'plot', 'dir'])
        cirrusfilter = CirrusCloudFilter(icefilter.result, bg_img=self.ir108,
                                         **cirrus_input)
        cirrusfilter.apply()
        self.add_mask(cirrusfilter.mask)

        # 5. Water cloud filtering
        water_input = self.get_kwargs(['ir108', 'vis008', 'nir016', 'vis006',
                                       'ir039', 'time', 'save', 'resize',
                                       'plot', 'dir'])
        waterfilter = WaterCloudFilter(cirrusfilter.result,
                                       cloudmask=cloudfilter.mask,
                                       bg_img=self.ir108,
                                       **water_input)
        waterfilter.apply()
        self.add_mask(waterfilter.mask)

        # 6. Spatial clustering
        self.clusters = self.get_cloud_cluster(self.mask)
        if self.plot:
            self.plot_clusters(self.save, self.dir)

        # 7. Calculate cloud top height if no CTH array is given
        if not hasattr(self, 'cth') or self.cth is None:
            cth_input = self.get_kwargs(['ir108', 'elev', 'time', 'dir',
                                         'plot', 'save'])
            cth_input['ccl'] = cloudfilter.ccl
            cth_input['cloudmask'] = self.mask
            cth_input['interpolate'] = True
            lcthalgo = LowCloudHeightAlgorithm(**cth_input)
            lcthalgo.run()
            cth = lcthalgo.result
        else:
            cth = self.cth
        # Apply cloud top height filter
        cthfilter = SpatialCloudTopHeightFilter(waterfilter.result,
                                                cth=cth,
                                                elev=self.elev,
                                                time=self.time,
                                                bg_img=self.ir108,
                                                dir=self.dir,
                                                save=self.save,
                                                plot=self.plot,
                                                resize=self.resize)
        cthfilter.apply()
        self.add_mask(cthfilter.mask)
        self.cluster_cth = np.ma.masked_where(self.mask, cthfilter.cth)

        # Recalculate clusters
        self.clusters = self.get_cloud_cluster(self.mask)

        # 8. Test spatial inhomogeneity
        stdevfilter = SpatialHomogeneityFilter(cthfilter.result,
                                               ir108=self.ir108,
                                               bg_img=self.ir108,
                                               clusters=self.clusters,
                                               time=self.time,
                                               dir=self.dir,
                                               save=self.save,
                                               plot=self.plot,
                                               resize=self.resize)
        stdevfilter.apply()
        self.add_mask(stdevfilter.mask)

        # 9. Apply cloud microphysical filter
        physic_input = self.get_kwargs(['cot', 'reff', 'time', 'save',
                                        'resize', 'plot', 'dir'])
        physicfilter = CloudPhysicsFilter(stdevfilter.result,
                                          bg_img=self.ir108,
                                          **physic_input)
        physicfilter.apply()
        self.add_mask(physicfilter.mask)

        # 10. Fog - low stratus cloud differentiation
        # Recalculate clusters
        self.clusters = self.get_cloud_cluster(self.mask)
        # Run low cloud model
        lowcloud_input = self.get_kwargs(['ir108', 'lwp', 'reff', 'elev',
                                          'time', 'save', 'resize', 'plot',
                                          'dir'])
        # Choose cluster computation method
        lowcloud_input['single'] = self.single

        lowcloudfilter = LowCloudFilter(physicfilter.result,
                                        cth=self.cluster_cth,
                                        clusters=self.clusters,
                                        bg_img=self.ir108, **lowcloud_input)
        lowcloudfilter.apply()
        self.add_mask(lowcloudfilter.mask)

        # Set results
        self.result = lowcloudfilter.result
        self.mask = self.mask

        # Compute separate products for validaiton
        # Get cloud mask
        self.vcloudmask = icefilter.mask | cirrusfilter.mask #| ~cloudfilter.mask
        # Extract cloud base and top heights products
        self.cbh = lowcloudfilter.cbh  # Cloud base height
        self.fbh = lowcloudfilter.fbh  # Fog base height
        self.lcth = cth  # Low cloud top height

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        ret = True
        return ret

    def get_cloud_cluster(self, mask, reduce=True):
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
        if result.ndim != 2 and reduce:
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
        """Get neighboring cloud free BT and elevation values of potential
        fog cloud clusters and compute cloud top height from maximum BT
        differences for fog cloud contaminated pixel in comparison to cloud
        free areas and their corresponding elevation using a constant
        atmospheric lapse rate
        """
        from collections import defaultdict
        result = defaultdict(list)

        logger.info("Calculating low clouds top heights")
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


class LowCloudHeightAlgorithm(BaseSatelliteAlgorithm):
    """This class provide an algorithm for low cloud top height determination.
    The method is based on satellite images and uses additionally a digital
    elevation map in the background.
    The algorithm requires a selection of different masked input arrays.
        - Infrared 10.8 channel for cloud top temperature extraction
        - Low cloud areas to find cloudy and cloud free areas
        - Cloud confidence level
        - Digital elevation map

    The height assignment is then a two step process:
        1. Derive cloud top height by margin terrain relief extraction,
           if possible.
        2. Get cloud top height by applying a constant lapse rate for remaining
           clouds with unassignable margin height

    Returns:
        Array with cloud top heights in [m]
    """
    def __init__(self, *args, **kwargs):
        super(LowCloudHeightAlgorithm, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'interpolate'):
            self.interpolate = False
        if not hasattr(self, 'method'):
            self.method = "nearest"
        if not hasattr(self, 'single'):
            self.single = False
        self.nlcthneg = 0

    def isprocessible(self):
        """Test runability here"""
        attrlist = ['ir108', 'cloudmask', 'ccl', 'elev']
        ret = []
        for attr in attrlist:
            if hasattr(self, attr):
                ret.append(True)
            else:
                ret.append(False)
                logger.warning("Missing input attribute: {}".format(attr))

        return all(ret)

    def procedure(self):
        """ Apply low cloud height algorithm to input arrays"""
        logger.info("Starting low cloud height assignment algorithm")
        # Get cloud top temperatures
        ctt = self.ir108
        # Prepare result arrays
        self.dz = np.empty(self.ir108.shape, dtype=np.float)
        self.cth = np.empty(self.ir108.shape, dtype=np.float)
        self.cth[:] = np.nan
        # Init stat variables
        self.ndem = 0
        self.nlapse = 0
        # Calculate cloud clusters
        if not hasattr(self, 'clusters'):
            self.clusters = self.get_cloud_cluster(self.cloudmask)
        # Execute pixel wise height detection in two steps
        if self.elev.shape == ctt.shape:
            for index, val in np.ndenumerate(self.clusters):
                if val == 0:
                    self.dz[index] = np.nan
                    continue
                # Get neighbor elevations
                zcenter, zneigh, zids = self.get_neighbors(self.elev,
                                                           index[0],
                                                           index[1])
                # Get neighbor entity values
                idcenter, idneigh, ids = self.get_neighbors(self.clusters,
                                                            index[0],
                                                            index[1],
                                                            mask=zids)
                # Get neighbor temperature values
                tcenter, tneigh, ids = self.get_neighbors(self.ir108,
                                                          i=index[0],
                                                          j=index[1],
                                                          mask=ids)
                # Get neighbor cloud confidence values
                cclcenter, cclneigh, ids = self.get_neighbors(self.ccl,
                                                              i=index[0],
                                                              j=index[1],
                                                              mask=ids)
                # 1. Get margin neighbor pixel
                idmargin = [i for i, x in enumerate(idneigh) if x == 0]
                if not idmargin:
                    self.dz[index] = np.nan
                    continue
                # 2. Check margin elevation for minimum relief
                zmargin = [zneigh[i] for i in idmargin]
                delta_z = max([zcenter] + zmargin) - min([zcenter] + zmargin)
                self.dz[index] = delta_z
                # 3. Find rising terrain from cloudy to margin pixels
                idrise = [i for i, x in enumerate(zmargin) if x > zcenter]
                zrise = [zmargin[i] for i in idrise]
                # 4. Test Pixel for DEM height extraction
                if delta_z >= 50 and idrise:
                    cthmargin = [zmargin[i] for i in idrise]
                    cth = np.mean(cthmargin)
                    self.ndem += 1
                else:
                    tmargin = [tneigh[i] for i in idmargin]
                    cclmargin = [cclneigh[i] for i in idmargin]
                    cthmargin = self.apply_lapse_rate(tcenter, tmargin,
                                                      zmargin)
                    cth = np.nanmean(cthmargin)
                    if not np.isnan(cth):
                        self.nlapse += 1
                self.cth[index] = cth
        # Interpolate height values
        if not np.all(np.isnan(self.cth)):
            logger.info("Perform low cloud height interpolation")
            if self.interpolate:  # Optional interpolation
                self.cth_result = self.interpol_cth(self.cth, self.cloudmask,
                                                    self.method)
            else:  # Default linear regression height estimation
                self.cth_result = self.linreg_cth(self.cth, self.cloudmask,
                                                  ctt, self.single)
        else:
            self.cth_result = self.cth
            logger.warning("No LCTH interpolated height estimation possible")
        # Set results
        self.result = self.cth_result
        self.mask = self.cloudmask

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        self.lcth_stats()
        if self.plot:
            # Overwrite plotrange with valid result array range
            self.plotrange = (np.nanmin(self.result), np.nanmax(self.result))
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize)
        return True

    def lcth_stats(self):
        self.algo_size = self.mask.size
        self.algo_num = np.nansum(~self.mask)
        self.cthnan = np.sum(np.isnan(self.cth[~self.mask]))
        self.cthassign = self.algo_num - self.cthnan
        self.resultnan = np.sum(np.isnan(self.result[~self.mask]))
        self.ninterp = self.cthnan - self.resultnan
        self.minheight = np.nanmin(self.result)
        self.meanheight = np.nanmean(self.result)
        self.maxheight = np.nanmax(self.result)

        logger.info("""LCTH algorithm results for {} \n
                    Array size:              {}
                    Valid cells:             {}
                    Assigend cells           {}
                      DEM extracted cells    {}
                      Lapse rate cells       {}
                    Interpolated cells       {}
                    Remaining NaN cells      {}
                    Excluded negative cells  {}
                    Min height:              {}
                    Mean height:             {}
                    Max height:              {}"""
                    .format(self.name,
                            self.algo_size, self.algo_num, self.cthassign,
                            self.ndem, self.nlapse, self.ninterp,
                            self.resultnan, self.nlcthneg, self.minheight,
                            self.meanheight, self.maxheight))

    def interpol_cth(self, cth, mask, method='nearest'):
        """Interpolate cth for given cloud clusters with scipy interpolation
        griddata method

        Args:
        cth (Numpy array): Array of computed heigh values with gaps
        mask (Numpy mask): Mask for valid cloud cluster pixels
        method (string): Interpolation method (nearest, linear or cubic)

        Returns:
            Numpy array with interpolated cloud top height values in unmasked
            areas
        """
        # Enumerate dimensions
        x = np.arange(0, cth.shape[1])
        y = np.arange(0, cth.shape[0])
        array = np.ma.masked_invalid(cth)
        # Get meshgrid
        xx, yy = np.meshgrid(x, y)
        # Remove masked values from grids
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newcth = cth[~array.mask]
        # Interpolate the gridded and masked data
        result = interpolate.griddata((x1, y1), newcth.ravel(),
                                      (xx, yy), method=method)
        if np.any(np.isnan(result)):
            logger.warning("LCTH algorithm interpolation created NaN values")
        # Set invalide values
        result[mask] = np.nan

        return result

    def linreg_cth(self, cth, mask, ctt, single=False):
        """Interpolate cth for given cloud clusters by linear regression with
        provided cloud top temperature data.

        Args:
        cth (Numpy array): Array of computed heigh values with gaps
        mask (Numpy mask): Mask for valid cloud cluster pixels
        ctt (Numpy array): Array of cloud top temperatures
        single (Bool): Boolean value for activating single cloud regressions

        Returns:
            Numpy array with interpolated cloud top height values in unmasked
            areas
        """
        result = deepcopy(cth)
        # Overall cloud cluster regression
        # Enumerate dimensions
        x = ctt[~mask & ~np.isnan(cth)]
        y = cth[~mask & ~np.isnan(cth)]
        result = self.apply_linear_regression(x, y, ctt, cth, result)
        if single:
            # Single cloud cluster regression
            for index in np.arange(0, np.nanmax(self.clusters)):
                clstindex = self.clusters == index
                ctt_c = ctt[clstindex]
                cth_c = cth[clstindex]
                mask_c = mask[clstindex]
                result_c = result[clstindex]
                if np.sum(np.isnan(cth_c)) == 0:
                    continue
                elif np.sum(~np.isnan(cth_c)) == 0:
                    continue
                # Enumerate dimensions
                x = ctt_c[~np.isnan(cth_c)]
                y = cth_c[~np.isnan(cth_c)]
                # Get slope and offset and apply regression
                result_c = self.apply_linear_regression(x, y, ctt_c,
                                                        cth_c,
                                                        result_c)
                result[clstindex] = result_c

        #self.plot_linreg(x, y, m, c, savedir)
        if np.any(np.isnan(result)):
            logger.warning("LCTH linear regression created NaN values")
        # Set invalide values
        result[mask] = np.nan

        # Save regression plot
        if hasattr(self, 'time'):
            ts = self.time
        else:
            ts = datetime.now()
        savedir = os.path.join(self.dir, self.name + '_linreg_' +
                               datetime.strftime(ts,
                                                 '%Y%m%d%H%M') + '.png')
        return result

    def apply_linear_regression(self, x, y, x_arr, y_arr, out):
        """ Simple method to derive slope and offset by linear regression"""
        # Rearrange line equation to  y = Ap  from y = mx  + c with p = [m , c]
        A = np.vstack([x, np.ones(len(x))]).T
        # Solve by leat square fitting.
        m, c = np.linalg.lstsq(A, y)[0]
        # Apply regression
        out[np.isnan(y_arr)] = m * x_arr[np.isnan(y_arr)] + c

        return(out)

    def get_neighbors(self, arr, i, j, nan=False, mask=None):
        """Get neighbor cells by simple array indexing

        Args:
        arr (Numpy array): 2d numpy array
        i, j (integer): x, y indices of selected cell
        nan (Boolean): Optional return of invalide neighbors
        mask (Boolean numpy array): Apply mask to neighboring cells

        Returns:
            Centered cell value, neighbor values and mask
        """
        shp = arr.shape
        i_min = i - 1
        if i_min < 0:
            i_min = 0
        i_max = i + 2
        if i_max >= shp[0]:
            i_max = shp[0]
        j_min = j - 1
        if j_min < 0:
            j_min = 0
        j_max = j + 2
        if j_max >= shp[1]:
            j_max = shp[1]
        # Copy array slice and convert to float type for Nan value support
        neighbors = np.copy(arr[i_min:i_max, j_min:j_max].astype(float))
        center = arr[i, j]
        neighbors[i - i_min, j - j_min] = np.nan
        if mask is not None:
            neighbors[mask] = np.nan
        # Create valid neighbor mask
        ids = np.zeros(neighbors.shape).astype(bool)
        ids[np.isnan(neighbors)] = True
        # Return optional only non nan values
        if not nan:
            return center, neighbors[~np.isnan(neighbors)], ids
        else:
            return center, neighbors, ids

    def apply_lapse_rate(self, tcc, tcf, zneigh, lrate=-0.0054):
        """Compute cloud top height with constant atmosphere temperature
        lapse rate.

        Args:
        tcc (float): Temperature of cloud contaminated pixel in K
        tcf (float): Temperature of cloud free margin pixel in K
        zneigh (float): Elevation of cloud free margin pixel in m
        lrate (float): Environmental temperature lapse rate in K/m

        Returns:
            bool: True if successful, False otherwise."""
        cth = zneigh + (tcc - tcf) / lrate
        # Remove negative height values
        if isinstance(cth, np.ndarray):
            cth[cth < 0] = np.nan
            if np.all(np.isnan(cth)):
                self.nlcthneg += 1
        else:
            if cth < 0:
                cth = np.nan
                self.nlcthneg += 1

        return cth

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

    def cell_neighbors(self, arr, i, j, d=1):
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
        ids = [[k, l] for k in irange for l in jrange
               if k != icell or l != jcell]
        center = arr[i, j]  # Get center cell value from additional array

        return center, neighbors, ids

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

    def plot_linreg(self, x, y, m, c, saveto=None):
        """ Plot result of linear regression for DEM and lapse rate extracted
        low cloud top height and cloud top temperatures"""
        plt.plot(x, y, '.')
        plt.plot(x, m * x + c)
        plt.title("Linear Regression for LCTH and CTT")
        plt.xlabel('Cloud top temperature [K]')
        plt.ylabel('Low cloud top height [m]')
        if saveto is None:
            plt.show()
        else:
            plt.savefig(saveto)


class PanSharpeningAlgorithm(BaseSatelliteAlgorithm):
    """This class provide an algorithm for pansharpening of satellite channels
    by an spatial high resolution panchromatic channel.
    The method is based on satellite images for different channels with low
    resolution and a pan chromatic channel with higher spatial resolution.
    The low resolution multispectral images are than resampled on the high
    resolution grid by different approaches.

    Requires:
        mspec    List of multispectral cahnnels as numpy arrays
        pan      Panchromatic channel as numpy array
        area     Area definition object (PyTroll-mpop class) for the
                 multispectral channels.
        panarea  Area definition object (PyTroll-mpop class) for the
                 panchromatic channel.

    Implemented approaches:

    Hill: Local correlation approach

    A window-based pan-sharpening technique using linear local regressions is
    described by Hill et al. (1999).
    The idea for the method is not to calculate one regression function for
    the whole scene but one regression function for the calculation of every
    single pixel. Each of these is based on the degraded panchromatic high
    resolution channel.

        References:
        Hill, J., Diemer, C., St ̈over, O., and Udelhoven, T.: A local corre-
        lation approach for the fusion of remote sensing data with spa-
        tial resolutions in forestry applications, Int. Arch. Photogramm.
        Remote Sens., 32, Part 7-4-3 W6, Valladolid, Spain, 3–4 June,
        1999.


    Returns:
        Pansharpened satellite channels as numpy arrays.
    """
    def __init__(self, *args, **kwargs):
        super(PanSharpeningAlgorithm, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'method'):
            self.method = "hill"

    def isprocessible(self):
        """Test runability here"""
        attrlist = ['mspec', 'pan', 'area', 'panarea']
        ret = []
        for attr in attrlist:
            if hasattr(self, attr):
                if attr == 'area' and not isinstance(self.area,
                                                     geometry.AreaDefinition):
                    raise ValueError("The are is not of type area definition")
                ret.append(True)
            else:
                ret.append(False)
                logger.warning("Missing input attribute: {}".format(attr))

        return all(ret)

    def procedure(self):
        """ Apply pansharpening algorithm to multispectral input arrays"""
        logger.info("Starting pansharpening algorithm for {} images"
                    .format(len(self.mspec)))

        # Resample pan channel to degraded resolution
        pan_quick = image.ImageContainerNearest(self.pan, self.panarea,
                                                radius_of_influence=50000)
        pan_shrp_quick = pan_quick.resample(self.area)
        self.pan_degrad = pan_shrp_quick.image_data

        # Loop over  multispectral channels
        
        # Set results
        self.result = self.pan_degrad
        self.mask = np.zeros(self.pan_degrad.shape)

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        if self.plot:
            # Overwrite plotrange with valid result array range
            self.plotrange = (np.nanmin(self.result), np.nanmax(self.result))
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize)
        return True

