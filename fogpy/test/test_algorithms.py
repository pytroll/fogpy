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
from fogpy.algorithms import DayFogLowStratusAlgorithm
from fogpy.algorithms import NightFogLowStratusAlgorithm
from fogpy.algorithms import LowCloudHeightAlgorithm
from fogpy.filters import CloudFilter
from fogpy.filters import SnowFilter
from fogpy.filters import IceCloudFilter
from fogpy.filters import CirrusCloudFilter
from fogpy.filters import WaterCloudFilter
from pyorbital.orbital import Orbital
from pyorbital import tlefile
# Test data array order:
# ir108, ir039, vis08, nir16, vis06, ir087, ir120, elev, cot, reff, cwp,
# lat, lon, cth
# Use indexing and np.dsplit(testdata, 13) to extract specific products

# Import test data
base = os.path.split(fogpy.__file__)
testfile = os.path.join(base[0], '..', 'etc', 'fog_testdata.npy')
testfile2 = os.path.join(base[0], '..', 'etc', 'fog_testdata2.npy')
testfile_night = os.path.join(base[0], '..', 'etc', 'fog_testdata_night.npy')
testfile_night2 = os.path.join(base[0], '..', 'etc', 'fog_testdata_night2.npy')
testdata = np.load(testfile)
testdata2 = np.load(testfile2)
testdata_night = np.load(testfile_night)
testdata_night2 = np.load(testfile_night2)


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


class Test_DayFogLowStratusAlgorithm(unittest.TestCase):

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
        # Load second test dataset
        inputs = np.dsplit(testdata2, 14)
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

        self.time2 = datetime(2014, 8, 27, 7, 15)

        self.input2 = {'vis006': self.vis006,
                       'vis008': self.vis008,
                       'ir108': self.ir108,
                       'nir016': self.nir016,
                       'ir039': self.ir039,
                       'ir120': self.ir120,
                       'ir087': self.ir087,
                       'lat': self.lat,
                       'lon': self.lon,
                       'time': self.time2,
                       'elev': self.elev,
                       'cot': self.cot,
                       'reff': self.reff,
                       'lwp': self.lwp,
                       'cth': self.cth,
                       'plot': True,
                       'save': True,
                       'dir': '/tmp/FLS',
                       'resize': '5'}

    def tearDown(self):
        pass

    def test_fls_algorithm(self):
        flsalgo = DayFogLowStratusAlgorithm(**self.input)
        ret, mask = flsalgo.run()
        self.assertEqual(flsalgo.ir108.shape, (141, 298))
        self.assertEqual(ret.shape, (141, 298))
        self.assertEqual(flsalgo.shape, (141, 298))
        self.assertEqual(np.ma.is_mask(flsalgo.mask), True)
        self.assertLessEqual(np.nanmax(flsalgo.cluster_cth), 2000)

    # Using other tset data set
    def test_fls_algorithm_other(self):
        flsalgo = DayFogLowStratusAlgorithm(**self.input2)
        ret, mask = flsalgo.run()
        self.assertEqual(flsalgo.ir108.shape, (141, 298))
        self.assertEqual(ret.shape, (141, 298))
        self.assertEqual(flsalgo.shape, (141, 298))
        self.assertEqual(np.ma.is_mask(flsalgo.mask), True)
        self.assertLessEqual(np.nanmax(flsalgo.cluster_cth), 2000)

    def test_fls_cth_tdiff(self):
        flsalgo = DayFogLowStratusAlgorithm(**self.input)
        # Prepare input for cluster height detection
        testshp = (5, 5)
        testmask = np.ma.make_mask(np.ones(testshp))
        cfmask = ~testmask
        cfmask[2, 2] = True
        ccmask = ~cfmask
        cluster = np.ma.masked_array(np.ones(testshp, dtype=np.int8),
                                     mask=ccmask)
        cf_arr = np.ma.masked_array(np.full(testshp, 280.), mask=cfmask)
        bt_cc = np.ma.masked_array(np.full(testshp, 250), mask=ccmask)
        elevation = np.full(testshp, 0)
        # Test height detection
        testcth = flsalgo.get_lowcloud_cth(cluster, cf_arr, bt_cc, elevation)
        comparecth = (280 - 250) / 0.65 * 100 - (0 - 0)
        # Evaluate results
        self.assertAlmostEqual(testcth[1][0], comparecth)

    def test_fls_cth_zdiff(self):
        flsalgo = DayFogLowStratusAlgorithm(**self.input)
        # Prepare input for cluster height detection
        testshp = (5, 5)
        testmask = np.ma.make_mask(np.ones(testshp))
        cfmask = ~testmask
        cfmask[2, 2] = True
        ccmask = ~cfmask
        cluster = np.ma.masked_array(np.ones(testshp, dtype=np.int8),
                                     mask=ccmask)
        cf_arr = np.ma.masked_array(np.full(testshp, 280.), mask=cfmask)
        bt_cc = np.ma.masked_array(np.full(testshp, 250), mask=ccmask)
        elevation = np.full(testshp, 2000)
        elevation[2, 2] = 0
        # Test height detection
        testcth = flsalgo.get_lowcloud_cth(cluster, cf_arr, bt_cc, elevation)
        comparecth = (280 - 250) / 0.65 * 100 - (2000 - 0)
        # Evaluate results
        self.assertAlmostEqual(testcth[1][0], comparecth)


class Test_LowCloudHeightAlgorithm(unittest.TestCase):

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

        # Remove nodata values (9999) from elevation
        self.elev[self.elev == 9999] = np.nan

        self.time = datetime(2013, 11, 12, 8, 30, 00)

        self.input = {'vis006': self.vis006,
                      'vis008': self.vis008,
                      'ir108': self.ir108,
                      'nir016': self.nir016,
                      'ir039': self.ir039,
                      'ir120': self.ir120,
                      'ir087': self.ir087,
                      'bg_img': self.ir108,
                      'lat': self.lat,
                      'lon': self.lon,
                      'time': self.time,
                      'elev': self.elev,
                      'cth': self.cth,
                      'plot': True,
                      'save': True,
                      'dir': '/tmp/FLS',
                      'resize': '5'}

        # Define two small artificial test data
        test_ir = np.array([[290., 290., 290.],
                            [290., 260., 290.],
                            [290., 290., 290.]])
        test_ccl = np.array([[0.2, 0.4, 0.1],
                             [0.3, 1, 0.2],
                             [0, 0.4, 0.1]])
        test_clusters = np.array([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 0]])
        test_cmask = np.array([[True, True, True],
                               [True, False, True],
                               [True, True, True]], dtype=bool)
        test_elev = np.array([[800., 900., 1000.],
                              [700., 500., 600.],
                              [400., 300., 200.]])
        test_elev2 = np.array([[800., 900., 1000.],
                               [700., 1500., 600.],
                               [400., 300., 200.]])
        self.testinput = {'ir108': test_ir,
                          'elev': test_elev,
                          'ccl': test_ccl,
                          'cloudmask': test_cmask}
        self.testinput2 = {'ir108': test_ir,
                           'elev': test_elev2,
                           'ccl': test_ccl,
                           'cloudmask': test_cmask}

    def tearDown(self):
        pass

    def test_lcth_algorithm_interpolate(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        cth = np.random.random_integers(0, 10, (5, 5)).astype(float)
        cth[cth > 7] = np.nan
        mask = np.random.random_integers(0, 1, (5, 5)).astype(bool)
        result = lcthalgo.interpol_cth(cth, mask)
        self.assertEqual(result.shape, (5, 5))
        self.assertTrue(np.all(np.isnan(result[mask])))
        self.assertGreaterEqual(np.sum(np.isnan(result)), np.sum(mask))

    def test_lcth_algorithm_linreg(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        cth = np.random.random_integers(0, 10, (5, 5)).astype(float)
        ctt = np.random.random_integers(260, 290, (5, 5)).astype(float)
        cth[cth > 7] = np.nan
        mask = np.random.random_integers(0, 1, (5, 5)).astype(bool)
        result = lcthalgo.linreg_cth(cth, mask, ctt)
        self.assertEqual(result.shape, (5, 5))
        self.assertTrue(np.all(np.isnan(result[mask])))
        self.assertEqual(np.sum(np.isnan(result)), np.sum(mask))

    def test_lcth_algorithm_nan_neighbor(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        elev = np.empty((3, 3))
        elev[:] = np.nan
        lcthalgo.elev = elev
        ret, mask = lcthalgo.run()
        self.assertEqual(lcthalgo.ir108.shape, (3, 3))
        self.assertEqual(ret.shape, (3, 3))
        self.assertEqual(lcthalgo.shape, (3, 3))
        self.assertEqual(np.ma.is_mask(lcthalgo.mask), True)
        self.assertEqual(np.isnan(np.nanmax(lcthalgo.dz)), True)
        self.assertEqual(np.isnan(np.nanmax(lcthalgo.cth)), True)

    def test_lcth_algorithm_real(self):
        # Compute cloud mask
        cloudfilter = CloudFilter(self.input['ir108'], **self.input)
        cloudfilter.apply()
        snowfilter = SnowFilter(cloudfilter.result, **self.input)
        snowfilter.apply()
        icefilter = IceCloudFilter(snowfilter.result, **self.input)
        icefilter.apply()
        cirrusfilter = CirrusCloudFilter(icefilter.result, **self.input)
        cirrusfilter.apply()
        waterfilter = WaterCloudFilter(cirrusfilter.result,
                                       cloudmask=cloudfilter.mask,
                                       **self.input)
        ret, mask = waterfilter.apply()

        self.ccl = cloudfilter.ccl
        self.input['ccl'] = self.ccl
        self.input['cloudmask'] = ret.mask

        lcthalgo = LowCloudHeightAlgorithm(**self.input)
        ret, mask = lcthalgo.run()
        self.assertEqual(lcthalgo.ir108.shape, (141, 298))
        self.assertEqual(ret.shape, (141, 298))
        self.assertEqual(lcthalgo.shape, (141, 298))
        self.assertEqual(np.ma.is_mask(lcthalgo.mask), True)
        self.assertLessEqual(round(np.nanmax(lcthalgo.dz), 2), 1524.43)

    def test_lcth_algorithm_artificial(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        ret, mask = lcthalgo.run()
        self.assertEqual(lcthalgo.ir108.shape, (3, 3))
        self.assertEqual(ret.shape, (3, 3))
        self.assertEqual(lcthalgo.shape, (3, 3))
        self.assertEqual(np.ma.is_mask(lcthalgo.mask), True)
        self.assertEqual(np.nanmax(lcthalgo.dz), 800.)
        self.assertEqual(np.nanmax(lcthalgo.cth), 800.)

    def test_lcth_algorithm_artificial_complement(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput2)
        ret, mask = lcthalgo.run()
        self.assertEqual(lcthalgo.ir108.shape, (3, 3))
        self.assertEqual(ret.shape, (3, 3))
        self.assertEqual(lcthalgo.shape, (3, 3))
        self.assertEqual(np.ma.is_mask(lcthalgo.mask), True)
        self.assertEqual(np.nanmax(lcthalgo.dz), 1300.)
        self.assertEqual(np.nanmax(np.around(lcthalgo.cth, 2)), 6168.06)

    def test_lcth_lapse_rate(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        test_z = lcthalgo.apply_lapse_rate(265, 270, 1000)
        test_z2 = lcthalgo.apply_lapse_rate(260, 290, 800)
        test_z3 = lcthalgo.apply_lapse_rate(260, 290, 1000)
        test_z4 = lcthalgo.apply_lapse_rate(290, np.array([290, 260, 300]),
                                            np.array([1000, 1500, 900]))
        test_z5 = lcthalgo.apply_lapse_rate(290, 260, 1000)
        self.assertEqual(round(test_z, 2), 1925.93)
        self.assertEqual(round(test_z2, 2), 6355.56)
        self.assertEqual(round(test_z3, 2), 6555.56)
        self.assertEqual(test_z4[0], 1000.)
        self.assertTrue(np.isnan(test_z4[1]))
        self.assertEqual(round(test_z4[2], 2), 2751.85)
        self.assertTrue(np.isnan(test_z5))

    def test_lcth_direct_neighbors(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        elev = self.testinput2['elev']
        center, neigh, ids = lcthalgo.get_neighbors(elev, 1, 1)
        compare = np.array([800., 900., 1000., 700., 600., 400., 300., 200.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper left corner
        center, neigh, ids = lcthalgo.get_neighbors(elev, 0, 0)
        compare = np.array([900., 700., 1500.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower left corner
        center, neigh, ids = lcthalgo.get_neighbors(elev, 2, 0)
        compare = np.array([700., 1500., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper right corner
        center, neigh, ids = lcthalgo.get_neighbors(elev, 0, 2)
        compare = np.array([900., 1500., 600.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower right corner
        center, neigh, ids = lcthalgo.get_neighbors(elev, 2, 2)
        compare = np.array([1500., 600., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test middle right cell
        center, neigh, ids = lcthalgo.get_neighbors(elev, 1, 2)
        compare = np.array([900., 1000., 1500., 300., 200.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test middle left cell
        center, neigh, ids = lcthalgo.get_neighbors(elev, 1, 0)
        compare = np.array([800., 900., 1500., 400., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper middle cell
        center, neigh, ids = lcthalgo.get_neighbors(elev, 0, 1)
        compare = np.array([800., 1000., 700., 1500., 600.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower middle cell
        center, neigh, ids = lcthalgo.get_neighbors(elev, 2, 1)
        compare = np.array([700., 1500., 600., 400., 200.])
        self.assertTrue(np.alltrue(compare == neigh))

    def test_lcth_cell_neighbors(self):
        lcthalgo = LowCloudHeightAlgorithm(**self.testinput)
        elev = self.testinput2['elev']
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 1, 1)
        compare = np.array([800., 900., 1000., 700., 600., 400., 300., 200.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper left corner
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 0, 0)
        compare = np.array([900., 700., 1500.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower left corner
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 2, 0)
        compare = np.array([700., 1500., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper right corner
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 0, 2)
        compare = np.array([900., 1500., 600.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower right corner
        center, neigh, ids = lcthalgo.get_neighbors(elev, 2, 2)
        compare = np.array([1500., 600., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test middle right cell
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 1, 2)
        compare = np.array([900., 1000., 1500., 300., 200.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test middle left cell
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 1, 0)
        compare = np.array([800., 900., 1500., 400., 300.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test upper middle cell
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 0, 1)
        compare = np.array([800., 1000., 700., 1500., 600.])
        self.assertTrue(np.alltrue(compare == neigh))
        # Test lower middle cell
        center, neigh, ids = lcthalgo.cell_neighbors(elev, 2, 1)
        compare = np.array([700., 1500., 600., 400., 200.])
        self.assertTrue(np.alltrue(compare == neigh))


class Test_NightFogLowStratusAlgorithm(unittest.TestCase):

    def setUp(self):
        # Load test data
        inputs = np.dsplit(testdata_night, 14)
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

        self.time = datetime(2013, 11, 12, 6, 00, 00)
        # METEOSAT-10 (MSG-3) TLE file from 01.08.2017
        # http://celestrak.com/NORAD/elements/weather.txt
        line1 = "1 38552U 12035B   17212.14216600 -.00000019  00000-0  00000-0 0  9998"
        line2 = "2 38552   0.8450 357.8180 0002245 136.4998 225.6885  1.00275354 18379"
        # Import TLE file
        self.tle = tlefile.read('meteosat 10', line1=line1, line2=line2)
        self.orbital = Orbital('meteosat 10', line1=self.tle.line1,
                               line2=self.tle.line2)
        # Compute satellite zenith angle
        azi, ele = self.orbital.get_observer_look(self.time, self.lon,
                                                  self.lat, self.elev)
        self.sza = ele

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
                      'resize': '5',
                      'sza': self.sza}

    def tearDown(self):
        pass

    def test_nightfls_algorithm(self):
        flsalgo = NightFogLowStratusAlgorithm(**self.input)
        ret, mask = flsalgo.run()
        flsalgo.plot_result()
        self.assertEqual(flsalgo.ir108.shape, (141, 298))
        self.assertEqual(ret.shape, (141, 298))
        self.assertEqual(flsalgo.shape, (141, 298))
        self.assertEqual(np.ma.is_mask(flsalgo.mask), True)

    def test_nightfls_turningpoints_with_valley(self):
        y = np.array([1, 2, 4, 7, 5, 2, 0, 2, 3, 4, 6])
        flsalgo = NightFogLowStratusAlgorithm(**self.input)
        tvalues, valleys = flsalgo.get_turningpoints(y)
        self.assertEqual(tvalues[5], True)
        self.assertEqual(tvalues[2], True)
        self.assertEqual(np.sum(tvalues), 2)
        self.assertEqual(np.alen(valleys), 1)
        self.assertEqual(valleys, 0)

    def test_nightfls_turningpoints_with_thres(self):
        y = np.array([1, 2, 4, 7, 5, 2, 0, 2, 3, 4, 6])
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        flsalgo = NightFogLowStratusAlgorithm(**self.input)
        tvalues, thres = flsalgo.get_turningpoints(y, x)
        self.assertEqual(tvalues[5], True)
        self.assertEqual(tvalues[2], True)
        self.assertEqual(np.sum(tvalues), 2)
        self.assertEqual(np.alen(thres), 1)
        self.assertEqual(thres, 7)

    def test_nightfls_turningpoints_no_valley(self):
        y = np.array([1, 2, 4, 7, 5, 2, 1, 1, 1, 0])
        flsalgo = NightFogLowStratusAlgorithm(**self.input)
        tvalues, valleys = flsalgo.get_turningpoints(y)
        self.assertEqual(tvalues[2], True)
        self.assertEqual(np.sum(tvalues), 1)
        self.assertEqual(np.alen(valleys), 0)

    def test_nightfls_slope(self):
        y = np.array([1, 2, 4, 7, 5, 2, 1, 1, 1, 0])
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        flsalgo = NightFogLowStratusAlgorithm(**self.input)
        slope, thres = flsalgo.get_slope(y, x)
        self.assertEqual(slope[2], 3)
        self.assertEqual(slope[8], -1)
        self.assertEqual(np.alen(slope), 9)
        self.assertEqual(thres, 6)


def suite():
    """The test suite for test_filter.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_BaseSatelliteAlgorithm))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_DayFogLowStratusAlgorithm))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_LowCloudHeightAlgorithm))

    return mysuite


if __name__ == "__main__":
    unittest.main()
