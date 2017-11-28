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

""" This module tests the array filter class """

import unittest
import numpy as np
import os
import fogpy

from datetime import datetime
from fogpy.filters import BaseArrayFilter
from fogpy.filters import CloudFilter
from fogpy.filters import SnowFilter
from fogpy.filters import IceCloudFilter
from fogpy.filters import CirrusCloudFilter
from fogpy.filters import WaterCloudFilter
from fogpy.filters import SpatialCloudTopHeightFilter
from fogpy.filters import SpatialHomogeneityFilter
from fogpy.filters import LowCloudFilter
from fogpy.filters import CloudMotionFilter
from fogpy.algorithms import DayFogLowStratusAlgorithm
from pyresample import geometry

# Test data array order:
# ir108, ir039, vis08, nir16, vis06, ir087, ir120, elev, cot, reff, cwp,
# lat, lon, cth
# Use indexing and np.dsplit(testdata, 13) to extract specific products

# Import test data
base = os.path.split(fogpy.__file__)
testfile = os.path.join(base[0], '..', 'etc', 'fog_testdata.npy')
testfile_pre = os.path.join(base[0], '..', 'etc', 'fog_testdata_pre.npy')
testfile2 = os.path.join(base[0], '..', 'etc', 'fog_testdata2.npy')
testdata = np.load(testfile)
testdata_pre = np.load(testfile_pre)
testdata2 = np.load(testfile2)

# Get area definition for test data
area_id = "geos_germ"
name = "geos_germ"
proj_id = "geos"
proj_dict = {'a': '6378169.00', 'lon_0': '0.00', 'h': '35785831.00',
             'b': '6356583.80', 'proj': 'geos', 'lat_0': '0.00'}
x_size = 298
y_size = 141
area_extent = (214528.82635591552, 4370087.2110124603,
               1108648.9697693815, 4793144.0573926577)
area_def = geometry.AreaDefinition(area_id, name, proj_id,
                                   proj_dict, x_size, y_size,
                                   area_extent)
# HRV size
x_size = 893
y_size = 422
hrvarea_def = geometry.AreaDefinition(area_id, name, proj_id,
                                      proj_dict, x_size, y_size,
                                      area_extent)


class Test_ArrayFilter(unittest.TestCase):

    def setUp(self):
        self.testarray = np.arange(0, 16, dtype=np.float).reshape((4, 4))
        self.testmarray = np.ma.array(self.testarray,
                                      mask=np.zeros(self.testarray.shape))

    def tearDown(self):
        pass

    def test_array_filter(self):
        newfilter = BaseArrayFilter(self.testarray)
        newfilter.attrlist = []
        ret, mask = newfilter.apply()
        self.assertEqual(newfilter.arr.shape, (4, 4))
        self.assertEqual(ret.shape, (4, 4))

    def test_marray_filter(self):
        newfilter = BaseArrayFilter(self.testmarray)
        newfilter.attrlist = []
        ret, mask = newfilter.apply()
        self.assertEqual(newfilter.arr.shape, (4, 4))
        self.assertEqual(ret.shape, (4, 4))
        self.assertEqual(np.ma.is_masked(ret), True)
        self.assertEqual(np.ma.is_mask(newfilter.inmask), True)

    def test_array_filter_param(self):
        param = {'test1': 'test1', 'test2': 0, 'test3': 0.1, 'test4': True}
        newfilter = BaseArrayFilter(self.testarray, **param)
        newfilter.attrlist = []
        ret, mask = newfilter.apply()
        self.assertEqual(len(newfilter.test1), 5)
        self.assertEqual(newfilter.test2, 0)
        self.assertEqual(type(newfilter.test3).__name__, 'float')
        self.assertEqual(newfilter.test4, True)
        self.assertEqual(newfilter.arr.shape, (4, 4))
        self.assertEqual(ret.shape, (4, 4))
        self.assertEqual(np.ma.is_mask(mask), True)


class Test_CloudFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        self.ir108, self.ir039 = np.dsplit(testdata, 14)[:2]
        self.input = {'ir108': self.ir108,
                      'ir039': self.ir039,
                      'bg_img': self.ir108}

    def tearDown(self):
        pass

    def test_cloud_filter(self):
        # Create cloud filter
        testfilter = CloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()
        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.ir039[20, 100], 269.573815979)
        self.assertAlmostEqual(testfilter.minpeak, -19.744686196671914)
        self.assertAlmostEqual(testfilter.maxpeak, 1.11645277953)
        self.assertAlmostEqual(testfilter.thres, -3.51935588185)
        self.assertEqual(np.sum(testfilter.mask), 20551)

    def test_cloud_filter_plot(self):
        # Create cloud filter
        testfilter = CloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()
        testfilter.plot_cloud_hist('/tmp/cloud_filter_hist_20131120830.png')
        testfilter.plot_filter(save=True)
        testfilter.plot_filter(save=True, area=area_def, type='tif')
        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.ir039[20, 100], 269.573815979)
        self.assertAlmostEqual(testfilter.minpeak, -19.744686196671914)
        self.assertAlmostEqual(testfilter.maxpeak, 1.11645277953)
        self.assertAlmostEqual(testfilter.thres, -3.51935588185)
        self.assertEqual(np.sum(testfilter.mask), 20551)

    def test_masked_cloud_filter(self):
        # Create cloud filter
        inarr = np.ma.masked_greater(self.input['ir108'], 275)
        testfilter = CloudFilter(inarr, **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.ir039[20, 100], 269.573815979)
        self.assertAlmostEqual(testfilter.minpeak, -19.744686196671914)
        self.assertAlmostEqual(testfilter.maxpeak, 1.11645277953)
        self.assertAlmostEqual(testfilter.thres, -3.51935588185)
        self.assertEqual(np.sum(testfilter.mask), 20551)
        self.assertEqual(np.sum(testfilter.inmask), 4653)
        self.assertEqual(testfilter.new_masked, 15922)

    def test_ccl_cloud_filter(self):
        # Create cloud filter
        testfilter = CloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(testfilter.ccl.squeeze().shape, (141, 298))
        self.assertAlmostEqual(round(testfilter.ccl[95, 276], 3), 0.544)
        self.assertAlmostEqual(round(testfilter.ccl[29, 216], 3), 1)
        self.assertAlmostEqual(round(testfilter.ccl[78, 45], 3), 0.303)
        self.assertAlmostEqual(round(testfilter.ccl[61, 261], 3), 0)
        self.assertEqual(round(np.nanmax(testfilter.cm_diff), 2), 3.43)
        self.assertTrue(all(testfilter.ccl[testfilter.cm_diff <
                                           testfilter.thres] > 0.5))
        self.assertTrue(all(testfilter.ccl[testfilter.cm_diff >
                                           testfilter.thres] < 0.5))
        self.assertEqual(np.nanmax(testfilter.ccl), 1)
        self.assertEqual(np.nanmin(testfilter.ccl), 0)


class Test_SnowFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 14)
        self.ir108 = inputs[0]
        self.ir039 = inputs[1]
        self.vis008 = inputs[2]
        self.nir016 = inputs[3]
        self.vis006 = inputs[4]
        self.ir087 = inputs[5]
        self.ir120 = inputs[6]
        self.elev = inputs[7]
        self.cot = inputs[8]
        self.reff = inputs[9]
        self.cwp = inputs[10]

        self.input = {'vis006': self.vis006,
                      'vis008': self.vis008,
                      'ir108': self.ir108,
                      'nir016': self.nir016}

    def tearDown(self):
        pass

    def test_snow_filter(self):
        # Create cloud filter
        testfilter = SnowFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.vis008[25, 100], 13.40515625)
        self.assertAlmostEqual(testfilter.ndsi[30, 214], 0.12547279)
        self.assertAlmostEqual(testfilter.ndsi[135, 170], 0.62573861)
        self.assertEqual(np.sum(testfilter.mask), 577)


class Test_IceCloudFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 14)
        self.ir108 = inputs[0]
        self.ir039 = inputs[1]
        self.vis008 = inputs[2]
        self.nir016 = inputs[3]
        self.vis006 = inputs[4]
        self.ir087 = inputs[5]
        self.ir120 = inputs[6]
        self.elev = inputs[7]
        self.cot = inputs[8]
        self.reff = inputs[9]
        self.cwp = inputs[10]
        self.lat = inputs[11]
        self.lon = inputs[12]
        self.cth = inputs[13]

        self.input = {'ir108': self.ir108,
                      'ir120': self.ir120,
                      'ir087': self.ir087}

    def tearDown(self):
        pass

    def test_ice_cloud_filter(self):
        # Create cloud filter
        testfilter = IceCloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.vis008[25, 100], 13.40515625)
        self.assertAlmostEqual(testfilter.ic_diff[50, 50], -0.91323156)
        self.assertAlmostEqual(testfilter.ic_diff[110, 70], 3.05561071)
        self.assertAlmostEqual(testfilter.ic_diff[126, 144], 3.05652842)
        self.assertEqual(np.sum(testfilter.mask), 36632)


class Test_CirrusCloudFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 14)
        self.ir108 = inputs[0]
        self.ir039 = inputs[1]
        self.vis008 = inputs[2]
        self.nir016 = inputs[3]
        self.vis006 = inputs[4]
        self.ir087 = inputs[5]
        self.ir120 = inputs[6]
        self.elev = inputs[7]
        self.cot = inputs[8]
        self.reff = inputs[9]
        self.cwp = inputs[10]
        self.lat = inputs[11]
        self.lon = inputs[12]
        self.cth = inputs[13]

        self.time = datetime(2013, 11, 12, 8, 30, 00)

        self.input = {'ir108': self.ir108,
                      'ir120': self.ir120,
                      'ir087': self.ir087,
                      'lat': self.lat,
                      'lon': self.lon,
                      'time': self.time}

    def tearDown(self):
        pass

    def test_cirrus_cloud_filter(self):
        # Create cloud filter
        testfilter = CirrusCloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.vis008[25, 100], 13.40515625)
        self.assertEqual(np.sum(testfilter.bt_ci_mask |
                                testfilter.strong_ci_mask),
                         np.sum(testfilter.mask))
        self.assertAlmostEqual(testfilter.bt_thres[50, 50], 1.1)
        self.assertGreater(testfilter.bt_diff[50, 50], testfilter.bt_thres[50,
                                                                           50])
        self.assertLess(testfilter.strong_ci_diff[110, 70], 0)
        self.assertLess(testfilter.bt_diff[110, 70], testfilter.bt_thres[50,
                                                                         50])
        self.assertEqual(np.sum(testfilter.mask), 9398)


class Test_WaterCloudFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 14)
        self.ir108 = inputs[0]
        self.ir039 = inputs[1]
        self.vis008 = inputs[2]
        self.nir016 = inputs[3]
        self.vis006 = inputs[4]
        self.ir087 = inputs[5]
        self.ir120 = inputs[6]
        self.elev = inputs[7]
        self.cot = inputs[8]
        self.reff = inputs[9]
        self.cwp = inputs[10]
        self.lat = inputs[11]
        self.lon = inputs[12]
        self.cth = inputs[13]

        self.time = datetime(2013, 11, 12, 8, 30, 00)

        # Create cloud mask
        testfilter = CloudFilter(self.ir108, ir108=self.ir108,
                                 ir039=self.ir039)
        ret, cloudmask = testfilter.apply()
        self.input = {'ir108': self.ir108,
                      'vis006': self.vis006,
                      'nir016': self.nir016,
                      'ir039': self.ir039,
                      'cloudmask': cloudmask}

    def tearDown(self):
        pass

    def test_water_cloud_filter(self):
        # Create cloud filter
        testfilter = WaterCloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        cloud_free_ma = np.ma.masked_where(~testfilter.cloudmask,
                                           testfilter.ir039)
        testmean = np.nanmean(cloud_free_ma[140, :])
        self.assertEqual(testfilter.lat_cloudfree[140], testmean)
        self.assertAlmostEqual(np.sum(~testfilter.cloudmask) +
                               np.sum(testfilter.cloudmask), 42018)
        self.assertAlmostEqual(np.sum(testfilter.cloudmask), 20551)
        self.assertEqual(np.sum(testfilter.mask), 19857)
        self.assertEqual(testfilter.line, 141)


class Test_SpatialCloudTopHeightFilter(unittest.TestCase):

    def setUp(self):
        # Define artificial test data with random cth around 1000 mean value
        self.ir108 = 1.0 * np.random.randn(10, 10) + 260
        self.elev = np.zeros((10, 10))
        self.cth = np.random.randn(10, 10) + 1000
        self.masksum = np.sum(self.cth > 1000)

    def tearDown(self):
        pass

    def test_spatial_cth_filter_artifical(self):
        # Create cloud filter
        testfilter = SpatialCloudTopHeightFilter(self.ir108,
                                                 cth=self.cth,
                                                 elev=self.elev)
        ret, mask = testfilter.apply()
        lowcth = np.ma.masked_where(mask, testfilter.cth)
        lowcth2 = np.ma.masked_where(testfilter.mask, testfilter.cth)
        # Evaluate results
        np.testing.assert_array_equal(ret, self.ir108)
        self.assertEqual(np.nansum(testfilter.mask), self.masksum)
        self.assertLessEqual(np.max(lowcth), 1000)
        self.assertLessEqual(np.max(lowcth2), 1000)


class Test_SpatialHomogeneityFilter(unittest.TestCase):

    def setUp(self):
        # Define artificial test data with low standard deviation
        flsalgo = DayFogLowStratusAlgorithm()
        self.low_sd_ir = 1.0 * np.random.randn(10, 10) + 260
        self.low_sd_mask = np.random.randint(2, size=(10, 10))
        self.low_sd_ir_masked = np.ma.masked_where(self.low_sd_mask,
                                                   self.low_sd_ir)
        self.low_sd_cluster = np.ma.masked_invalid(np.ones((10, 10)))
        self.low_sd_clusterma = flsalgo.get_cloud_cluster(self.low_sd_mask)
        # Define artificial test data with high standard deviation
        self.high_sd_ir = 20.0 * np.random.randn(10, 10) + 260
        self.high_sd_mask = np.random.randint(2, size=(10, 10))
        self.high_sd_ir_masked = np.ma.masked_where(self.high_sd_mask,
                                                    self.high_sd_ir)
        self.high_sd_cluster = np.ma.masked_invalid(np.ones((10, 10)))
        self.high_sd_clusterma = flsalgo.get_cloud_cluster(self.high_sd_mask)

    def tearDown(self):
        pass

    def test_spatial_homogenity_filter_nomask_low(self):
        # Create cloud filter
        testfilter = SpatialHomogeneityFilter(self.low_sd_ir,
                                              ir108=self.low_sd_ir,
                                              clusters=self.low_sd_cluster)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.nansum(testfilter.mask), 0)

    def test_spatial_homogenity_filter_nomask_high(self):
        # Create cloud filter
        testfilter = SpatialHomogeneityFilter(self.high_sd_ir,
                                              ir108=self.high_sd_ir,
                                              clusters=self.high_sd_cluster)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.nansum(testfilter.mask), 100)
        self.assertEqual(np.nansum(testfilter.inmask), 0)

    def test_spatial_homogenity_filter_lowsd(self):
        # Create cloud filter
        testfilter = SpatialHomogeneityFilter(self.low_sd_ir_masked,
                                              ir108=self.low_sd_ir_masked,
                                              clusters=self.low_sd_clusterma)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertNotEqual(np.nansum(testfilter.mask), 0)
        self.assertEqual(np.nansum(testfilter.mask), np.nansum(self.low_sd_mask))

    def test_spatial_homogenity_filter_highsd(self):
        # Create cloud filter
        testfilter = SpatialHomogeneityFilter(self.high_sd_ir_masked,
                                              ir108=self.high_sd_ir_masked,
                                              clusters=self.high_sd_clusterma)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertNotEqual(np.nansum(testfilter.mask), 0)
        self.assertEqual(np.nansum(testfilter.mask), 100)


class Test_LowCloudFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 14)
        self.ir108 = inputs[0]
        self.ir039 = inputs[1]
        self.vis008 = inputs[2]
        self.nir016 = inputs[3]
        self.vis006 = inputs[4]
        self.ir087 = inputs[5]
        self.ir120 = inputs[6]
        self.elev = inputs[7]
        self.cot = inputs[8]
        self.reff = inputs[9]
        self.lwp = inputs[10]
        self.lat = inputs[11]
        self.lon = inputs[12]
        self.cth = inputs[13]

        self.time = datetime(2013, 11, 12, 8, 30, 00)
        self.input = {'vis006': self.vis006,
                      'vis008': self.vis008,
                      'ir108': self.ir108,
                      'nir016': self.nir016,
                      'ir039': self.ir039,
                      'ir120': self.ir120,
                      'ir087': self.ir087,
                      'lat': self.lat,
                      'lon': self.lon,
                      'time': self.time,
                      'elev': self.elev,
                      'cot': self.cot,
                      'reff': self.reff,
                      'lwp': self.lwp,
                      'cth': self.cth,
                      'plot': True,
                      'save': True,
                      'dir': '/tmp/FLS',
                      'resize': '5'}
        # Create cloud top heights and clusters
        from fogpy.algorithms import DayFogLowStratusAlgorithm as FLS
        # Calculate cloud and snow masks
        cloudfilter = CloudFilter(self.ir108, **self.input)
        ret, self.cloudmask = cloudfilter.apply()
        snowfilter = SnowFilter(cloudfilter.result, **self.input)
        ret, snowmask = snowfilter.apply()
        # Get clusters
        fls = FLS(**self.input)
        self.clusters = FLS.get_cloud_cluster(fls, self.cloudmask, False)
        self.clusters.mask[self.clusters != 120] = True
        self.input = {'ir108': self.ir108,
                      'lwp': self.lwp,
                      'reff': self.reff,
                      'elev': self.elev,
                      'clusters': self.clusters,
                      'cth': self.cth}

        # Define small artificial test data
        dim = (2, 2, 1)
        test_ir = np.empty(dim)
        test_ir.fill(260)
        lwp_choice = np.array([0.01, 0.1, 1, 10, 100, 1000])
        test_lwp_choice = np.random.choice(lwp_choice, dim)
        test_lwp_static = np.empty(dim)
        test_lwp_static.fill(94)
        test_reff = np.empty(dim)
        test_reff.fill(10e-6)
        test_cmask = np.empty(dim)
        test_cmask.fill(0)
        test_clusters = np.empty(dim)
        test_clusters.fill(1)
        test_clusters = np.ma.masked_array(test_clusters, test_cmask)
        test_elev = np.empty(dim)
        test_elev.fill(100)
        test_cth = np.empty(dim)
        test_cth.fill(200)
        # Testdata for ground cloud fog
        self.test_lwp1 = {'ir108': test_ir,
                          'lwp': test_lwp_static,
                          'reff': test_reff,
                          'elev': test_elev,
                          'clusters': test_clusters,
                          'cth': test_cth}
        # Init with big droplet radius
        # Increase in droplet radius prevend declaration as ground fog
        test_reff = np.empty(dim)
        test_reff.fill(20e-5)
        self.test_lwp2 = {'ir108': test_ir,
                          'lwp': test_lwp_static,
                          'reff': test_reff,
                          'elev': test_elev,
                          'clusters': test_clusters,
                          'cth': test_cth}
        # Randomly choose liquid water paths
        test_reff = np.empty(dim)
        test_reff.fill(10e-6)  # Reset radius
        self.test_lwp3 = {'ir108': test_ir,
                          'lwp': test_lwp_choice,
                          'reff': test_reff,
                          'elev': test_elev,
                          'clusters': test_clusters,
                          'cth': test_cth}

    def tearDown(self):
        pass

    def test_lowcloud_filter_clusters(self):
        # Create cloud filter
        testfilter = LowCloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.sum(self.cloudmask), 20551)
        self.assertEqual(np.nanmax(len(testfilter.result_list)), 1)

    def test_lowcloud_filter_single(self):
        # Create cloud filter
        input_single = self.input
        input_single['single'] = True
        input_single['substitude'] = True
        testfilter = LowCloudFilter(input_single['ir108'], **input_single)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.sum(self.cloudmask), 20551)
        self.assertEqual(np.nanmax(len(testfilter.result_list)), 16)
        self.assertEqual(np.sum(testfilter.fog_mask), 42018)
        self.assertEqual(np.sum(testfilter.mask), 42018)

    def test_lowcloud_filter_single_lwp_allfog(self):
        # Create cloud filter
        input_single = self.test_lwp1
        input_single['single'] = True
        input_single['substitude'] = False
        testfilter = LowCloudFilter(input_single['ir108'], **input_single)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.nanmax(len(testfilter.result_list)), 4)
        self.assertEqual(np.sum(testfilter.fog_mask), 0)
        self.assertEqual(np.sum(testfilter.mask), 0)

    def test_lowcloud_filter_single_lwp_nofog(self):
        # Create cloud filter
        input_single = self.test_lwp2
        input_single['single'] = True
        input_single['substitude'] = False
        testfilter = LowCloudFilter(input_single['ir108'], **input_single)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.nanmax(len(testfilter.result_list)), 4)
        self.assertEqual(np.sum(testfilter.fog_mask), 4)
        self.assertEqual(np.sum(testfilter.mask), 4)

    def test_lowcloud_filter_single_lwp_randomfog(self):
        # Create cloud filter
        input_single = self.test_lwp3
        input_single['single'] = True
        input_single['substitude'] = False
        testfilter = LowCloudFilter(input_single['ir108'], **input_single)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertEqual(np.nanmax(len(testfilter.result_list)), 4)
        nfog = np.sum(input_single['lwp'] <= 10)
        self.assertEqual(np.sum(testfilter.fog_mask), nfog)
        self.assertEqual(np.sum(testfilter.mask), nfog)


class Test_CloudMotionFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        self.ir108 = np.dsplit(testdata, 14)[0]
        self.preir108 = np.dsplit(testdata_pre, 14)[0]
        print(type(self.ir108))
        print(type(self.preir108))

    def tearDown(self):
        pass

    def test_cloud_motion_filter(self):
        # Create cloud filter
        testfilter = CloudMotionFilter(self.ir108,
                                       ir108=self.ir108,
                                       preir108=self.preir108)
        ret, mask = testfilter.apply()
        # Evaluate results
        self.assertEqual(ret, self.ir108)


def suite():
    """The test suite for test_filter.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_ArrayFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_CloudFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_SnowFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_IceCloudFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_CirrusCloudFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_WaterCloudFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_SpatialHomogeneityFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_CloudMotionFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_LowCloudFilter))

    return mysuite

if __name__ == "__main__":
    unittest.main()
