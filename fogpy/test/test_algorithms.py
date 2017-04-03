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

""" This module tests the satellite algorithm classes """

import fogpy
import numpy as np
import os
import unittest

from datetime import datetime
from fogpy.algorithms import BaseSatelliteAlgorithm
from fogpy.algorithms import FogLowStratusAlgorithm

# Test data array order:
# ir108, ir039, vis08, nir16, vis06, ir087, ir120, elev, cot, reff, cwp,
# lat, lon
# Use indexing and np.dsplit(testdata, 13) to extract specific products

# Import test data
base = os.path.split(fogpy.__file__)
testfile = os.path.join(base[0], '..', 'etc', 'fog_testdata.npy')
testdata = np.load(testfile)


class Test_BaseSatelliteAlgorithm(unittest.TestCase):

    def setUp(self):
        self.testarray = np.arange(0, 16, dtype=np.float).reshape((4, 4))
        self.testmarray = np.ma.array(self.testarray,
                                      mask=np.ones(self.testarray.shape))
        self.input = {'test1': self.testarray, 'test2': self.testmarray}

    def tearDown(self):
        pass

    def test_base_algorithm(self):
        newalgo = BaseSatelliteAlgorithm(**self.input)
        ret, mask = newalgo.run()
        self.assertEqual(newalgo.test1.shape, (4, 4))
        self.assertEqual(np.ma.is_masked(newalgo.test2), True)
        self.assertEqual(ret.shape, (4, 4))
        self.assertEqual(newalgo.shape, (4, 4))


class Test_FogLowStratusAlgorithm(unittest.TestCase):

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

        self.input = {'vis006': self.vis006,
                      'vis008': self.vis008,
                      'ir108': self.ir108,
                      'nir016': self.nir016,
                      'ir039': self.ir039,
                      'ir120': self.ir120,
                      'ir087': self.ir087,
                      'lat': self.lat,
                      'lon': self.lon,
                      'time': self.time}

    def tearDown(self):
        pass

    def test_fls_algorithm(self):
        flsalgo = FogLowStratusAlgorithm(**self.input)
        ret, mask = flsalgo.run()
        flsalgo.plot_result()
        self.assertEqual(flsalgo.ir108.shape, (141, 298, 1))
        self.assertEqual(ret.shape, (141, 298, 1))
        self.assertEqual(flsalgo.shape, (141, 298, 1))
        self.assertEqual(np.ma.is_mask(flsalgo.mask), True)


def suite():
    """The test suite for test_filter.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_BaseSatelliteAlgorithm))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_FogLowStratusAlgorithm))

    return mysuite

if __name__ == "__main__":
    unittest.main()
