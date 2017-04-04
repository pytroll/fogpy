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

# Test data array order:
# ir108, ir039, vis08, nir16, vis06, ir087, ir120, elev, cot, reff, cwp,
# lat, lon
# Use indexing and np.dsplit(testdata, 13) to extract specific products

# Import test data
base = os.path.split(fogpy.__file__)
testfile = os.path.join(base[0], '..', 'etc', 'fog_testdata.npy')
testdata = np.load(testfile)


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
        self.ir108, self.ir039 = np.dsplit(testdata, 13)[:2]
        self.input = {'ir108': self.ir108,
                      'ir039': self.ir039}

    def tearDown(self):
        pass

    def test_cloud_filter(self):
        # Create cloud filter
        testfilter = CloudFilter(self.input['ir108'], **self.input)
        ret, mask = testfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.ir039[20, 100], 269.573815979)
        self.assertAlmostEqual(testfilter.minpeak, -8.7346406259)
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
        self.assertAlmostEqual(testfilter.minpeak, -8.7346406259)
        self.assertAlmostEqual(testfilter.maxpeak, 1.11645277953)
        self.assertAlmostEqual(testfilter.thres, -3.51935588185)
        self.assertEqual(np.sum(testfilter.mask), 20551)
        self.assertEqual(np.sum(testfilter.inmask), 4653)
        self.assertEqual(testfilter.new_masked, 15922)


class Test_SnowFilter(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata, 13)
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
        inputs = np.dsplit(testdata, 13)
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
        inputs = np.dsplit(testdata, 13)
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
        inputs = np.dsplit(testdata, 13)
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

    return mysuite

if __name__ == "__main__":
    unittest.main()
