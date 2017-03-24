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
from fogpy.filters import BaseArrayFilter
from fogpy.filters import CloudFilter

# Test data array order:
# ir108, ir039, vis08, nir16, vis06, ir087, ir120, elev, cot, reff, cwp
# Use indexing and np.dsplit(testdata, 11) to extract specific products

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
        ret, mask = newfilter.apply()
        self.assertEqual(newfilter.arr.shape, (4, 4))
        self.assertEqual(ret.shape, (4, 4))

    def test_marray_filter(self):
        newfilter = BaseArrayFilter(self.testmarray)
        ret, mask = newfilter.apply()
        self.assertEqual(newfilter.arr.shape, (4, 4))
        self.assertEqual(ret.shape, (4, 4))
        self.assertEqual(np.ma.is_masked(ret), True)
        self.assertEqual(np.ma.is_mask(newfilter.inmask), True)

    def test_array_filter_param(self):
        param = {'test1': 'test1', 'test2': 0, 'test3': 0.1, 'test4': True}
        newfilter = BaseArrayFilter(self.testarray, **param)
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
        self.ir108, self.ir039 = np.dsplit(testdata, 11)[:2]
        self.input = {'ir108': self.ir108,
                      'ir039': self.ir039}

    def tearDown(self):
        pass

    def test_cloud_filter(self):
        # Create cloud filter
        cloudfilter = CloudFilter(self.input['ir108'], **self.input)
        ret, mask = cloudfilter.apply()

        # Evaluate results
        self.assertAlmostEqual(self.ir108[0, 0], 244.044000086)
        self.assertAlmostEqual(self.ir039[20, 100], 269.573815979)
        self.assertAlmostEqual(cloudfilter.minpeak, -8.7346406259)
        self.assertAlmostEqual(cloudfilter.maxpeak, 1.11645277953)
        self.assertAlmostEqual(cloudfilter.thres, -3.51935588185)
        self.assertEqual(np.sum(cloudfilter.mask), 20551)


def suite():
    """The test suite for test_filter.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_ArrayFilter))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_CloudFilter))

    return mysuite

if __name__ == "__main__":
    unittest.main()
