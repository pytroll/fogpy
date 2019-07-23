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

"""This module implements an base satellite algorithm class

The implementation is based on the following publications:

    * Cermak, J., & Bendix, J. (2011). Detecting ground fog from space–a
      microphysics-based approach. International Journal of Remote Sensing,
      32(12), 3345-3371. doi:10.1016/j.atmosres.2007.11.009
    * Cermak, J., & Bendix, J. (2007). Dynamical nighttime fog/low stratus
      detection based on Meteosat SEVIRI data: A feasibility study. Pure and
      applied Geophysics, 164(6-7), 1179-1192. doi:10.1007/s00024-007-0213-8
    * Cermak, J., & Bendix, J. (2008). A novel approach to fog/low
      stratus detection using Meteosat 8 data. Atmospheric Research,
      87(3-4), 279-292. doi:10.1016/j.atmosres.2007.11.009
    * Cermak, J. (2006). SOFOS-a new satellite-based operational fog
      observation scheme. (PhD thesis), Philipps-Universität Marburg,
      Marburg, Germany. doi:doi.org/10.17192/z2006.0149

"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from copy import deepcopy
from datetime import datetime
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import measurements
from scipy.stats import linregress
from scipy import interpolate
from scipy import spatial
from .filters import CloudFilter
from .filters import SnowFilter
from .filters import IceCloudFilter
from .filters import CirrusCloudFilter
from .filters import WaterCloudFilter
from .filters import SpatialCloudTopHeightFilter
from .filters import SpatialHomogeneityFilter
from .filters import CloudPhysicsFilter
from .filters import LowCloudFilter
from pyresample import image, geometry
from pyresample.utils import generate_nearest_neighbour_linesample_arrays

logger = logging.getLogger(__name__)


class NotProcessibleError(Exception):
    """Exception to be raised when a filter is not applicable."""
    pass


class DummyException(Exception):
    """Never raised
    """


class BaseSatelliteAlgorithm(object):
    """This super filter class provide all functionalities to run an algorithm
    on satellite image arrays and return a new array as result."""
    def __init__(self, **kwargs):
        self.mask = None
        self.result = None
        self.attributes = []
        if kwargs is not None:
            for key, value in kwargs.items():
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
        """Start the algorithm and return results."""
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
        """Check processed algorithm for plausible results."""
        if self.plot:
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize)
        return True

    def add_mask(self, mask):
        """Compute the new array mask as union of all input array masks
        and computed masks."""
        if not np.ma.is_mask(mask):
            raise TypeError("Mask type is invalid")
        if self.mask is not None:
            self.mask = self.mask | mask
        else:
            self.mask = mask

    def get_kwargs(self, keys):
        """Return dictionary with passed keyword arguments."""
        return({key: self.__getattribute__(key) for key in self.attributes
                if key in keys})

    def plot_result(self, array=None, save=False, dir="/tmp", resize=1,
                    name='array', type='png', area=None, floating_point=False):
        """Plotting the algorithm result.

        Method disabled pending https://github.com/gerritholl/fogpy/issues/6"""

        # The problem is that instead of Image we should use XRImage, but
        # we're getting a masked array from downstream rather than an
        # xarray.DataArray

        raise NotImplementedError(
                "plot_result is disabled pending conversion to "
                "xarray/dask, see "
                "https://github.com/gerritholl/fogpy/issues/6")
        # Using Trollimage if available, else matplotlib is used to plot
        try:
            from trollimage.image import Image
            from trollimage.colormap import ylorrd
            from mpop.imageo.geo_image import GeoImage
        except ImportError:
            logger.info("{} results can't be plotted to: {}". format(self.name,
                                                                     dir))
            return 0
        if area is None:
            try:
                area = self.area
            except DummyException:
                Warning("Area object not found. Plotting filter result as"
                        " image")
                type = 'png'
        # Create image from data
        if array is None:
            if np.nanmax(self.result) > 1:
                self.plotrange = (np.nanmin(self.result),
                                  np.nanmax(self.result))
            if type == 'tif':
                result_img = GeoImage(self.result.squeeze(), area,
                                      self.time,
                                      mode="L")
            else:
                result_img = Image(self.result.squeeze(), mode='L',
                                   fill_value=None)
        else:
            self.plotrange = (np.nanmin(array), np.nanmax(array))
            if type == 'tif':
                result_img = GeoImage(array.squeeze(), area,
                                      self.time,
                                      mode="L")
            else:
                result_img = Image(array.squeeze(), mode='L', fill_value=None)
        result_img.stretch("crude")
        # Colorize image
        ylorrd.set_range(*self.plotrange)
        logger.info("Set color range to {}".format(self.plotrange))
        if not floating_point:
            result_img.colorize(ylorrd)
        if array is None:
            shape = self.result.shape
        else:
            shape = array.shape
        result_img.resize((shape[0] * int(resize),
                           shape[1] * int(resize)))
        if save:
            # Get output directory and image name
            if array is None:
                outname = self.name
            else:
                outname = self.name + '_' + name
            savedir = os.path.join(dir, outname + '_' +
                                   datetime.strftime(self.time,
                                                     '%Y%m%d%H%M') +
                                   '.' + type)
            if type == 'tif':
                result_img.save(savedir, floating_point=floating_point)
            else:
                result_img.save(savedir)
            logger.info("{} results are plotted to: {}". format(self.name,
                                                                self.dir))
        else:
            result_img.show()

        return(result_img)

    def check_dimension(self, arr):
        """Check and convert arrays to 2D."""
        if arr.ndim != 2:
            try:
                result = arr.squeeze()  # Try to reduce dimension
            except DummyException:
                raise ValueError("need 2-D input")
        else:
            result = arr

        return(result)

    def plot_clusters(self, save=False, dir="/tmp"):
        """Plot the cloud clusters."""
        # Get output directory and image name
        name = self.__class__.__name__
        savedir = os.path.join(dir, name + '_clusters_' +
                               datetime.strftime(self.time,
                                                 '%Y%m%d%H%M') + '.png')
        # Using Trollimage if available, else matplotlib is used to plot
        try:
            from trollimage.image import Image
            from trollimage.colormap import rainbow
        except ImportError:
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

    def plot_linreg(self, x, y, m, c, saveto=None, xlabel='x', ylabel='y',
                    title='Regression plot'):
        """ Plot result of linear regression for DEM and lapse rate extracted
        low cloud top height and cloud top temperatures."""
        plt.plot(x, y, '.')
        plt.plot(x, m * x + c)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if saveto is None:
            plt.show()
        else:
            plt.savefig(saveto)


class DayFogLowStratusAlgorithm(BaseSatelliteAlgorithm):
    """This algorithm implements a fog and low stratus detection and forecasting
    for geostationary satellite images from the SEVIRI instrument onboard of
    METEOSAT second generation MSG satellites. Seven MSG channels from the
    solar and infrared spectra are used. Therefore the algorithm is applicable
    for daytime scenes only.
    It is utilizing the methods proposed in different innovative studies:

    The implementation is based on the following publications:

        * Cermak, J., & Bendix, J. (2011). Detecting
          ground fog from space–a microphysics-based
          approach. International Journal of Remote Sensing, 32(12),
          3345-3371. doi:10.1016/j.atmosres.2007.11.009
        * Cermak, J., & Bendix, J. (2008). A novel approach to fog/low
          stratus detection using Meteosat 8 data. Atmospheric Research,
          87(3-4), 279-292. doi:10.1016/j.atmosres.2007.11.009
        * Cermak, J. (2006). SOFOS-a new satellite-based operational fog
          observation scheme. (PhD thesis), Philipps-Universität Marburg,
          Marburg, Germany. doi:doi.org/10.17192/z2006.0149

    The algorithm can be applied to satellite zenith angle lower than 70°
    and a maximum solar zenith angle of 80°.

    The algorithm workflow is a succession of differnt masking approaches
    from coarse to finer selection to find fog and low stratus clouds within
    provided satellite images. Afterwards a separation between fog and low
    clouds are made by calibrating a cloud base height with a low cloud model
    to satellite retrieval information. Then a fog dissipation and subsequently
    a nowcasting of fog can be done.

    Args:
        | ir108 (:obj:`ndarray`): Array for the 10.8 μm channel.
        | ir039 (:obj:`ndarray`): Array for the 3.9 μm channel.
        | vis008 (:obj:`ndarray`): Array for the 0.8 μm channel.
        | nir016 (:obj:`ndarray`): Array for the 1.6 μm channel.
        | vis006 (:obj:`ndarray`): Array for the 0.6 μm channel.
        | ir087 (:obj:`ndarray`): Array for the 8.7 μm channel.
        | ir120 (:obj:`ndarray`): Array for the 12.0 μm channel.
        | time (:obj:`datetime`): Datetime object for the satellite scence.
        | lat (:obj:`ndarray`): Array of latitude values.
        | lon (:obj:`ndarray`): Array of longitude values.
        | elevation (:obj:`ndarray`): Array of area elevation.
        | cot (:obj:`ndarray`): Array of cloud optical thickness (depth).
        | reff (:obj:`ndarray`): Array of cloud particle effective radius.
        | lwp (:obj:`ndarray`): Array of cloud liquid water path.

    Returns:
        Infrared image with fog mask

    Todo:
        ============================================ =====================
        Task description                             Implemented (yes/no):
        ============================================ =====================
        1.  Cloud masking                            yes

        2.  Snow masking                             yes

        3.  Ice cloud masking                        yes

        4.  Thin cirrus masking                      yes

        5.  Watercloud test                          yes

        6.  Spatial clustering                       yes

        7.  Maximum margin elevation                 yes

        8.  Surface homogenity check                 yes

        9.  Microphysics plausibility check          yes

        10.  Differenciate fog - low status          yes

        11.  Fog dissipation                         No

        12.  Fog Nowcasting                          No
        ============================================ =====================
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
        """ Apply different filters and low cloud model to input data."""
        logger.info("Starting fog and low cloud detection algorithm"
                    " in daytime mode")
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
        logger.info("Finish fog and low cloud detection algorithm")
        self.result = lowcloudfilter.result
        self.mask = self.mask

        # Compute separate products for validaiton
        # Get cloud mask
        self.vcloudmask = icefilter.mask | cirrusfilter.mask
        # Extract cloud base and top heights products
        self.cbh = lowcloudfilter.cbh  # Cloud base height
        self.fbh = lowcloudfilter.fbh  # Fog base height
        self.lcth = cth  # Low cloud top height

        return True

    def check_results(self):
        """Check processed algorithm for plausible results."""
        ret = True
        return ret

    @classmethod
    def get_cloud_cluster(self, mask, reduce=True):
        """ Enumerate low water cloud clusters by spatial vicinity.

        A mask is provided and the non masked values are spatially clustered
        using scipy label method

        Returns: Array with enumerated clusters
        """
        logger.info("Clustering low clouds")
        # Enumerate fog cloud clusters
        cluster = measurements.label(~mask.astype('bool'))
        # Get 10.8 channel sampled by the previous fog filters
        result = np.ma.masked_where(mask, cluster[0])
        # Check dimension
        if result.ndim != 2 and reduce:
            try:
                result = result.squeeze()  # Try to reduce dimension
            except DummyException:
                raise ValueError("need 2-D input")
        logger.debug("Number of spatial coherent fog cloud clusters: %s"
                     % np.nanmax(np.unique(result)))

        return result

    def get_lowcloud_cth(self, cluster, cf_arr, bt_cc, elevation):
        """Get neighboring cloud free BT and elevation values of potential
        fog cloud clusters and compute cloud top height from maximum BT
        differences for fog cloud contaminated pixel in comparison to cloud
        free areas and their corresponding elevation using a constant
        atmospheric lapse rate.
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

    Args:
        | ir108 (:obj:`ndarray`): Array of infrared window channel.
        | cloudmask (:obj:`MaskedArray`): Mask for cloud clusters.
        | ccl (:obj:`ndarray`): Array of cloud confidence level.
        | elev (:obj:`ndarray`): Array of elevation information.

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
        if not hasattr(self, 'plottype'):
            self.plottype = 'png'
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
        """ Apply low cloud height algorithm to input arrays."""
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
                # 4. Test Pixel for DEM height extraction
                if delta_z >= 50 and idrise:
                    cthmargin = [zmargin[i] for i in idrise]
                    cth = np.mean(cthmargin)
                    self.ndem += 1
                else:
                    tmargin = [tneigh[i] for i in idmargin]
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
        """Check processed algorithm for plausible results."""
        self.lcth_stats()
        if self.plot:
            # Overwrite plotrange with valid result array range
            self.plotrange = (np.nanmin(self.result), np.nanmax(self.result))
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize,
                             type=self.plottype)
        return True

    def lcth_stats(self):
        """Print out algorithm results to stdout."""
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
            | cth (:obj:`ndarray`): Array of computed heigh values with gaps.
            | mask (:obj:`MaskedArray`): Mask for valid cloud cluster pixels.
            | method (:obj:`str`): Interpolation method (nearest, linear or
                                   cubic).

        Returns:
            Numpy array with interpolated cloud top height values in unmasked
            areas.
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
            | cth (:obj:`ndarray`): Array of computed heigh values with gaps.
            | mask (:obj:`MaskedArray`): Mask for valid cloud cluster pixels.
            | ctt (:obj:`ndarray`): Array of cloud top temperatures.
            | single (:obj:`bool`): Boolean value for activating single cloud
                             regressions.

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

#         self.plot_linreg(x, y, m, c, savedir, 'Cloud top temperature [K]',
#                          'Low cloud top height [m]',
#                          'Linear Regression for LCTH and CTT')
        if np.any(np.isnan(result)):
            logger.warning("LCTH linear regression created NaN values")
        # Set invalide values
        result[mask] = np.nan

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
            | arr (:obj:`ndarray`): 2d numpy array.
            | i, j (:obj:`int`): x, y indices of selected cell.
            | nan (:obj:`ndarray`): Optional return of invalide neighbors.
            | mask (:obj:`MaskedArray`): Apply mask to neighboring cells.

        Returns:
            Centered cell value, neighbor values and mask.
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
            | tcc (:obj:`float`): Temperature of cloud contaminated pixel in K.
            | tcf (:obj:`float`): Temperature of cloud free margin pixel in K.
            | zneigh (:obj:`float`): Elevation of cloud free margin pixel in m.
            | lrate (:obj:`float`): Environmental temperature lapse rate, K/m.

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
            except DummyException:
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
            except DummyException:
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
        neighbors = [w[ix, jx][k, m] for k in irange for m in jrange
                     if k != icell or m != jcell]
        ids = [[k, m] for k in irange for m in jrange
               if k != icell or m != jcell]
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
            except DummyException:
                raise ValueError("need 2-D input")
        logger.debug("Number of spatial coherent fog cloud clusters: %s"
                     % np.nanmax(np.unique(result)))

        return result


class PanSharpeningAlgorithm(BaseSatelliteAlgorithm):
    """This class provide an algorithm for pansharpening of satellite channels
    by an spatial high resolution panchromatic channel.
    The method is based on satellite images for different channels with low
    resolution and a pan chromatic channel with higher spatial resolution.
    The low resolution multispectral images are than resampled on the high
    resolution grid by different approaches.

    Implemented approaches:

    *Hill - Local correlation approach*

    A window-based pan-sharpening technique using linear local regressions is
    described by Hill et al. (1999).
    The idea for the method is not to calculate one regression function for
    the whole scene but one regression function for the calculation of every
    single pixel. Each of these is based on the degraded panchromatic high
    resolution channel.

        * Hill, J., Diemer, C., Stover, O., and Udelhoven, T.: A local corre-
          lation approach for the fusion of remote sensing data with spa-
          tial resolutions in forestry applications, Int. Arch. Photogramm.
          Remote Sens., 32, Part 7-4-3 W6, Valladolid, Spain, 3–4 June,
          1999.

    Args:
        | mspec (:obj:`list`): List of multispectral channels as numpy
                                  arrays.
        | pan (:obj:`ndarray`): Panchromatic channel as numpy array.
        | area (:obj:`areadefinition`) Area definition object
                                       (pyresample class)
                                       for the multispectral channels.
        | panarea (:obj:`areadefinition`): Area definition object
                                           (pyresample class) for the
                                           panchromatic channel.

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
        # Convert multispetral channel option to list if required
        if not isinstance(self.mspec, list):
            self.mspec = [self.mspec]

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
        # Calculate row and column indices to translate multispectral channels
        # to  panchromatic
        panrow, pancol = generate_nearest_neighbour_linesample_arrays(
                self.area, self.panarea, 50000)

        self.result = []
        self.eval = []

        # Loop over  multispectral channels
        for chn in self.mspec:
            # Prepare empty pansharpened array
            panshrp_chn = np.empty(self.pan.shape)
            logger.info("Sharpen {} image to {}".format(chn.shape,
                                                        self.pan.shape))
            if self.method == 'hill':
                self.apply_hill_sharpening(chn, panrow, pancol, panshrp_chn)
        # Set results
        self.mask = np.zeros(self.pan_degrad.shape)

        return True

    def check_results(self):
        """Check processed algorithm for plausible results"""
        if self.plot:
            # Overwrite plotrange with valid result array range
            n = 0
            for chn in self.result:
                self.plotrange = (np.nanmin(chn), np.nanmax(chn))
                self.plot_result(array=chn, save=self.save, dir=self.dir,
                                 resize=self.resize,
                                 name='output_{}'.format(n),
                                 type='tif', area=self.panarea)
                self.plot_result(array=self.mspec[n], save=self.save,
                                 dir=self.dir, resize=self.resize,
                                 name='input_{}'.format(n), type='tif',
                                 area=self.area)
                self.plot_result(array=self.eval[n], save=self.save,
                                 dir=self.dir, resize=self.resize,
                                 name='eval_{}'.format(n), type='tif',
                                 area=self.area, floating_point=True)
                n += 1
        return True

    def apply_linear_regression(self, x, y):
        """ Simple method to derive slope and offset by linear regression"""
        # Rearrange line equation to  y = Ap  from y = mx  + c with p = [m , c]
        A = np.vstack([x, np.ones(len(x))]).T
        # Solve by leat square fitting.
        results, resids, rank, s = np.linalg.lstsq(A, y)
        m = results[0]
        c = results[1]
        mean = np.mean(y)
        # Calculate coefficient of determination
        var = np.sum((np.square(y - mean)))
        rsqrt = 1 - (resids / var)

#         if rsqrt < 0.2:
#             logger.warning("Computed low qualitiy regression R²: {}"
#                            .format(rsqrt))

        return(m, c, rsqrt, mean)

    def apply_hill_sharpening(self, chn, panrow, pancol, output):
        """ Local regresssion based pansharpening algorithm by HILL et al.
        1999. The nearest neighbor search is based on the scipy KDtree
        implementation, whereas remapping is done with pyresample methods.
        """
        logger.info("Using local regression approach by Hill")
        # Setup KDtree for nearest neighbor search
        indices = np.indices(chn.shape)
        tree = spatial.KDTree(list(zip(indices[0].ravel(),
                                       indices[1].ravel())))
        # Track progress
        todo = chn.size
        ready = 1
        # Array of evaluation criteria
        eval_array = np.empty(chn.shape)

        # Compute global regression
        gm, gc, grsqrt, gmean = self.apply_linear_regression(
                chn.ravel(), self.pan_degrad.ravel())

        # Loop over channel array
        for index, val in np.ndenumerate(chn):
            row = index[0]
            col = index[1]
            # Query tree for neighbors
            queryresult = tree.query(np.array([[row, col]]), k=25)
            neighs = tree.data[queryresult[1][0][1:]]
            # Get channel values for neighbors
            chn_neigh = chn[tuple(neighs.T)].squeeze()
            # Get panchromatic channel values for neighbors
            pan_neigh = self.pan_degrad[tuple(neighs.T)].squeeze()
            m, c, rsqrt, mean = self.apply_linear_regression(chn_neigh,
                                                             pan_neigh)
            # Apply local regression to panchromatic channel
            # Get values matching selected degaded pixel
            panvalues = self.pan[(panrow == row) & (pancol == col)]
            # Apply linear regression to cell selection
            if rsqrt >= 0.66:  # Condition for regression application
                panvalues_corr = c + panvalues * m
            else:   # Otherwith apply average value
                panvalues_corr = gc + panvalues * gm
            # Add corrected values to pansharpening channel output
            output[(panrow == row) & (pancol == col)] = panvalues_corr
            # Write coefficient of determination to evaluation array
            try:
                eval_array[index] = rsqrt
            except DummyException:
                Warning("Local linear regression not possible at: {}, {}"
                        .format(row, col))
            # Log tasks
            ready, todo = self.progressbar(ready, todo, chn.size)

        # Add pansharpend channel to result
        logger.info("Append pansharpened channel to result list...")
        self.result.append(output)
        self.eval.append(eval_array)

        return(output)

    def progressbar(self, ready, todo, size):
        """ simple method for printing a progress bar to stdout"""
        s = ('<' + (ready//(size//50)*'#') + (todo//(size//50)*'-') +
             '> ') + str(ready) + (' / {}'.format(size))
        print('\r'+s)
        todo -= 1
        ready += 1
        return(ready, todo)

    def get_dist_threshold(self, value, distance):
        # Calculate distributions and corresponding thresholds
        dist = self.get_bt_dist(value, distance)
        # Get turning points
        tvalues, valleyx = self.get_turningpoints(dist[0], dist[1][:-1])
        # Test modality of frequency distribution
        if np.alen(valleyx) == 1:
            thres = valleyx[0]  # Bimodal distribution valley point
        elif np.alen(valleyx) == 0:
            # Use point of slope declination
            slope, thres = self.get_slope(dist[0], dist[1][:-1])
        else:
            thres = np.nan
#             raise ValueError("Unknown form of distribution")
        return(thres)

    def get_sza_in_range(self, value, range):
        """Method to compute number of satellite zenith angles in given range
        around a value
        """
        mask = np.logical_and(self.sza > (value - range),
                              self.sza <= (value + range))
        count = mask.sum()
        return(count)

    def get_bt_dist(self, value, range):
        """Method to compute brightness temperature difference distribution
        for given range of satellite zenith angles.
        """
        mask = np.logical_and(self.sza > (value - range),
                              self.sza <= (value + range))
        if isinstance(self.bt_diff, np.ma.masked_array):
            v = self.bt_diff[mask].compressed()
        else:
            v = self.bt_diff[mask]
        return np.histogram(v)

    def plot_bt_hist(self, hist, saveto=None):
        plt.bar(hist[1][:-1], hist[0])
        plt.title("Histogram with 'auto' bins")
        if saveto is None:
            plt.show()
        else:
            plt.savefig(saveto)

    def plot_thres(self, saveto=None):
        plt.plot(self.sza, self.thres, 'ro')
        if self.slope and self.intercept:
            plt.plot(self.sza, self.slope * self.sza + self.intercept, 'b-')
        plt.title("Thresholds for different satellite zenith angles")
        plt.xlabel('Satellite zenith angle')
        plt.ylabel('BT difference threshold [K]')
        if saveto is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveto, self.name + "_" +
                                     datetime.strftime(self.time,
                                                       '%Y%m%d%H%M') +
                                     "_btthres.png"))

    def get_turningpoints(self, y, x=None):
        """Calculate turning points of bimodal histogram data and extract
        values for valleys or corresponding x-locations"""
        dx = np.diff(y)
        tvalues = dx[1:] * dx[:-1] < 0
        # Extract valley ids
        valley_ids = np.where(np.logical_and(dx[1:] > 0, tvalues))[0]
        valleys = y[valley_ids + 1]
        if x is not None:
            thres = x[valley_ids + 1]
            return(tvalues, thres)
        else:
            return(tvalues, valleys)

    def get_slope(self, y, x):
        """ Compute the slope of a one dimensional array"""
        slope = np.diff(y) / np.diff(x)
        decline = slope[1:] * slope[:-1]
        # Point of slope declination
        thres_id = np.where(np.logical_and(slope[1:] < 0, decline > 0))[0]
        if len(thres_id) != 0:
            thres = np.min(x[thres_id + 2])
        else:
            thres = None
        return(slope, thres)


class NightFogLowStratusAlgorithm(BaseSatelliteAlgorithm):
    """Night time fog and low stratus algorithm class.

    This algorithm implements a fog and low stratus detection and forecasting
    for geostationary satellite images from the SEVIRI instrument onboard of
    METEOSAT second generation MSG satellites for night time scenes.
    Two infrared MSG channels are used. Therefore the algorithm is applicable
    for day and night times in general, but is optimized for night time usage.
    It is utilizing the methods proposed in the following study:

        * Cermak, J., & Bendix, J. (2007). Dynamical nighttime
          fog/low stratus detection based on Meteosat SEVIRI data:
          A feasibility study. Pure and applied Geophysics, 164(6-7),
          1179-1192. doi:10.1007/s00024-007-0213-8

    The emissivity differences between the two window channels 10.8 and 3.9
    are considerably large for small droplets (effective radius = 4 um) in
    comparison to large droplets (effective radius of 10 um). Therefore the
    temperature differnce for these two channels can be utilized to detect
    small droplet clouds.
    Due to the large spectral width of SEVIRI middle infrared channel (3.9 um)
    and partly overlap of CO2 absorption bands the distinction between clear
    and FLS areas are hampered. The level of absorption influence vary with
    season and latitude.

    Hence the detection algorithm is based on the following approaches:

        * Dynamic determination of temperature difference thresholds for the
          given scene.
        * Consideration of local CO2 absorption variations by
          satellite-zenith-angel-specific threshold determination.

    Args:
        | chn108 (:obj:`ndarray`): Array for the 10.8 μm channel.
        | chn39 (:obj:`ndarray`): Array for the 3.9 μm channel.
        | sza (:obj:`ndarray`): Array for the satellite viewing zenith angle.
        | elev (:obj:`ndarray`): Array for the altitude.
        | time (:obj:`datetime`): Datetime object for the satellite scence.
        | lat (:obj:`ndarray`): Array of latitude values.
        | lon (:obj:`ndarray`): Array of longitude values.

    Returns:
        Infrared image with fog mask.

    Todo:
        ============================================ =====================
        Task description                             Implemented (yes/no):
        ============================================ =====================
        1.  Calculate temperature difference         yes

        2.  Retrieve localized difference thresholds yes

        3.  Smooth thresholds                        yes

        4.  Apply result thresholds                  yes

        5.  Derive confidence level                  yes
        ============================================ =====================
     """
    def __init__(self, *args, **kwargs):
        super(NightFogLowStratusAlgorithm, self).__init__(*args, **kwargs)
        # Set additional class attribute
        if not hasattr(self, 'minrange'):
            self.minrange = 0.5  # Minimum range for SZA calculation
        if not hasattr(self, 'trange'):
            self.trange = 500  # Target range for SZA calculation
        if not hasattr(self, 'fcr'):
            self.fcr = 2  # Fog confidence range in K

    def isprocessible(self):
        """Test runability here."""
        attrlist = ['ir108', 'ir039', 'lat', 'lon', 'time', 'sza']
        ret = []
        for attr in attrlist:
            if hasattr(self, attr):
                ret.append(True)
            else:
                ret.append(False)
                logger.warning("Missing input attribute: {}".format(attr))

        return all(ret)

    def procedure(self):
        """ Run nighttime fog and low stratus detection scheme."""
        logger.info("Starting fog and low cloud detection algorithm"
                    " in nighttime mode")
        #######################################################################
        # 1. Calculate temperature difference
        # Get differences
        self.bt_diff = self.ir108 - self.ir039
        logger.info("Satellite infrared differences in the range of {} - {}"
                    .format(np.nanmin(self.bt_diff),
                            np.nanmax(self.bt_diff)))
        # Comupte minimum and maximum satellite zenith angles
        minsza = np.nanmin(self.sza)
        maxsza = np.nanmax(self.sza)
        logger.info("Satellite zenith angles in the range of {} - {}"
                    .format(minsza, maxsza))
        #######################################################################
        # 2. Retrieve localized difference thresholds - Minimisin CO2 effects
        # Vectorize methods
        self.vget_sza_in_range = np.vectorize(self.get_sza_in_range)
        self.vget_bt_dist = np.vectorize(self.get_bt_dist)
        self.vget_dist_threshold = np.vectorize(self.get_dist_threshold)
        # Define starting distance
        distance = self.minrange
        # Get range partitions
        szarange = np.arange(np.nanmin(self.sza), np.nanmax(self.sza),
                             distance)
        nsza = self.vget_sza_in_range(szarange, distance)

        while np.min(nsza) < self.trange:
            distance += 0.5
            logger.info("Testing SZA distance: {} for target range: {}"
                        .format(distance, self.trange))
            szarange = np.arange(np.nanmin(self.sza), np.nanmax(self.sza),
                                 distance)
            nsza = self.vget_sza_in_range(szarange, distance)
        logger.info("Calibrated SZA range: {} for n: {} values in range"
                    .format(distance, self.trange))
        self.distance = distance
        # Calculate distributions and corresponding thresholds
        thresrange = self.vget_dist_threshold(szarange, distance)
        self.thres = np.empty(self.sza.shape)
        self.thres[:] = np.nan
        for i in np.arange(len(szarange)):
            if i != len(szarange) - 1:
                thresmask = np.logical_and(self.sza >= szarange[i],
                                           self.sza < szarange[i + 1])
                self.thres[thresmask] = thresrange[i]
        logger.info("Calculated thresholds in the range of {} - {}"
                    .format(np.nanmin(self.thres),
                            np.nanmax(self.thres)))

        #######################################################################
        # 3. Smooth thresholds
        nanmask = np.isnan(self.thres)  # Create mask for nan values
        # Fit linear regression to regional thresholds
        linreg = linregress(self.sza[~nanmask].ravel(),
                            self.thres[~nanmask].ravel())
        self.slope, self.intercept, rval, pval, stderr = linreg
        logger.info("Fitted linear regression threshold function with "
                    "slope: {} and intercept: {}"
                    .format(round(self.slope, 4),
                            round(self.intercept, 4)))

        #######################################################################
        # 4. Apply thresholds to infrared channel difference
        self.thres_linreg = self.slope * self.sza + self.intercept
        self.flsmask = self.bt_diff < self.thres_linreg

        #######################################################################
        # 5. Compute FLS confidence level
        self.flsconflvl = (self.thres_linreg - self.bt_diff - self.fcr) \
            / (-2 * self.fcr)
        # Set max and min confidence levels to range: 1 - 0
        self.flsconflvl[self.flsconflvl > 1] = 1
        self.flsconflvl[self.flsconflvl < 0] = 0
        # Set results
        logger.info("Finish fog and low cloud detection algorithm")
        self.result = np.ma.masked_array(self.ir108, self.flsmask)
        self.mask = self.flsmask

        return True

    def check_results(self):
        """Check processed algorithm for plausible results."""
        if self.plot:
            self.plot_result(save=self.save, dir=self.dir, resize=self.resize)
            self.plot_result(array=self.flsconflvl, save=self.save,
                             dir=self.dir, resize=self.resize,
                             name="confidlvl")
            if self.save:
                self.plot_thres(saveto=self.dir)
            else:
                self.plot_thres()

        return True

    def get_dist_threshold(self, value, distance):
        """Calculate thresholds for distribution of given sza value range."""
        # Calculate distributions and corresponding thresholds
        dist = self.get_bt_dist(value, distance)
        # Get turning points
        tvalues, valleyx = self.get_turningpoints(dist[0], dist[1][:-1])
        # Test modality of frequency distribution
        if np.alen(valleyx) == 1:
            thres = valleyx[0]  # Bimodal distribution valley point
        elif np.alen(valleyx) == 0:
            # Use point of slope declination
            slope, thres = self.get_slope(dist[0], dist[1][:-1])
        else:
            thres = np.nan
#             raise ValueError("Unknown form of distribution")
        return(thres)

    def get_sza_in_range(self, value, range):
        """Method to compute number of satellite zenith angles in given range
        around a value.
        """
        mask = np.logical_and(self.sza >= value,
                              self.sza < (value + range))
        count = mask.sum()
        return(count)

    def get_bt_dist(self, value, range):
        """Method to compute brightness temperature difference distribution
        for given range of satellite zenith angles.
        """
        mask = np.logical_and(self.sza >= value,
                              self.sza < (value + range))
        if isinstance(self.bt_diff, np.ma.masked_array):
            v = self.bt_diff[mask].compressed()
        else:
            v = self.bt_diff[mask]
        return np.histogram(v)

    def plot_bt_hist(self, hist, saveto=None):
        """Plot histogram of temperature distribution."""
        plt.bar(hist[1][:-1], hist[0])
        plt.title("Histogram with 'auto' bins")
        if saveto is None:
            plt.show()
        else:
            plt.savefig(saveto)

    def plot_thres(self, saveto=None):
        """Potting of satellite-zenith-angles specific thresholds."""
        plt.plot(self.sza, self.thres, 'ro')
        if self.slope and self.intercept:
            plt.plot(self.sza, self.slope * self.sza + self.intercept, 'b-')
        plt.title("Thresholds for different satellite zenith angles")
        plt.xlabel('Satellite zenith angle')
        plt.ylabel('BT difference threshold [K]')
        if saveto is None:
            plt.show()
        else:
            plt.savefig(os.path.join(saveto, self.name + "_" +
                                     datetime.strftime(self.time,
                                                       '%Y%m%d%H%M') +
                                     "_btthres.png"))

    def get_turningpoints(self, y, x=None):
        """Calculate turning points of bimodal histogram data and extract
        values for valleys or corresponding x-locations."""
        dx = np.diff(y)
        tvalues = dx[1:] * dx[:-1] < 0
        # Extract valley ids
        valley_ids = np.where(np.logical_and(dx[1:] > 0, tvalues))[0]
        valleys = y[valley_ids + 1]
        if x is not None:
            thres = x[valley_ids + 1]
            return(tvalues, thres)
        else:
            return(tvalues, valleys)

    def get_slope(self, y, x):
        """ Compute the slope of a one dimensional array."""
        slope = np.diff(y) / np.diff(x)
        decline = slope[1:] * slope[:-1]
        # Point of slope declination
        thres_id = np.where(np.logical_and(slope[1:] < 0, decline > 0))[0]
        if len(thres_id) != 0:
            thres = np.min(x[thres_id + 2])
        else:
            thres = None
        return(slope, thres)
