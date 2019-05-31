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

""" This module test the low cloud water class """

import unittest
import numpy as np
from fogpy.lowwatercloud import LowWaterCloud
from fogpy.lowwatercloud import CloudLayer


class Test_LowWaterCloud(unittest.TestCase):

    def setUp(self):
        self.lwc = LowWaterCloud(2000., 255., 400., 0, 10e-6)
        self.thinlwc = LowWaterCloud(2000., 255., 1., 0, 10e-6)
        self.nanlwc = LowWaterCloud(np.nan, 255., 400., 0, 10e-6)
        self.nodatalwc = LowWaterCloud(2000., 255., -9999, 0, 10e-6)

    def tearDown(self):
        pass

    def test_get_sat_vapour_pressure_buck(self):
        psv_m50 = self.lwc.get_sat_vapour_pressure(-50)
        psv_m20 = self.lwc.get_sat_vapour_pressure(-20)
        psv_0 = self.lwc.get_sat_vapour_pressure(0)
        psv_20 = self.lwc.get_sat_vapour_pressure(20)
        psv_50 = self.lwc.get_sat_vapour_pressure(50)
        self.assertAlmostEqual(psv_m50, 0.064, 3)
        self.assertAlmostEqual(psv_m20, 1.256, 3)
        self.assertAlmostEqual(psv_0, 6.112, 3)
        self.assertAlmostEqual(psv_20, 23.383, 3)
        self.assertAlmostEqual(psv_50, 123.494, 3)

    def test_get_sat_vapour_pressure_magnus(self):
        psv_m50 = self.lwc.get_sat_vapour_pressure(-50, 'magnus')
        psv_m20 = self.lwc.get_sat_vapour_pressure(-20, 'magnus')
        psv_0 = self.lwc.get_sat_vapour_pressure(0, 'magnus')
        psv_20 = self.lwc.get_sat_vapour_pressure(20, 'magnus')
        psv_50 = self.lwc.get_sat_vapour_pressure(50, 'magnus')
        self.assertAlmostEqual(psv_m50, 0.064, 3)
        self.assertAlmostEqual(psv_m20, 1.254, 3)
        self.assertAlmostEqual(psv_0, 6.108, 3)
        self.assertAlmostEqual(psv_20, 23.420, 3)
        self.assertAlmostEqual(psv_50, 123.335, 3)

    def test_get_air_pressure(self):
        pa_1 = self.lwc.get_air_pressure(610)
        pa_2 = self.lwc.get_air_pressure(0)
        self.assertAlmostEqual(pa_1, 942.08, 2)
        self.assertAlmostEqual(pa_2, 1013.25, 2)

    def test_get_moist_adiabatic_lapse_temp(self):
        temp_0 = self.lwc.get_moist_adiabatic_lapse_temp(500, 1500, 0)
        self.assertAlmostEqual(temp_0, 6.5, 1)

    def test_get_vapour_mixing_ratio(self):
        vmr = self.lwc.get_vapour_mixing_ratio(1007.26, 8.44)
        self.assertAlmostEqual(vmr, 5.26, 2)

    def test_get_cloud_based_vapour_mixing_ratio(self):
        self.lwc.cbh = 0
        cb_vmr = self.lwc.get_cloud_based_vapour_mixing_ratio()
        self.assertAlmostEqual(cb_vmr, 2.57, 2)

    def test_get_liquid_mixing_ratio(self):
        self.lwc.cbh = 0
        self.lwc.get_cloud_based_vapour_mixing_ratio()

        temp = self.lwc.get_moist_adiabatic_lapse_temp(0, self.lwc.cth,
                                                       self.lwc.ctt, True)
        cb_press = self.lwc.get_air_pressure(self.lwc.cbh)
        psv = self.lwc.get_sat_vapour_pressure(temp, self.lwc.vapour_method)
        vmr = self.lwc.get_vapour_mixing_ratio(cb_press, psv)
        lmr = self.lwc.get_liquid_mixing_ratio(self.lwc.cb_vmr, vmr)
        self.assertAlmostEqual(self.lwc.cb_vmr, vmr)
        self.assertAlmostEqual(lmr, 0)

    def test_get_liquid_water_content(self):
        lwc = LowWaterCloud(2000., 255., 400., 0)
        lwc = self.lwc.get_liquid_water_content(1950, 2000, 1.091, 1.392, 0.0,
                                                1.518, 50)
        self.assertAlmostEqual(lwc, 1.519, 3)

    def test_get_liquid_water_path(self):
        self.lwc.init_cloud_layers(421., 100)
        self.lwc.get_liquid_water_path()
        lwc = LowWaterCloud(2000., 255., 400., 0)
        cl = CloudLayer(1900, 2000, lwc)
        lwc.get_liquid_water_path()
        self.assertAlmostEqual(len(lwc.layers), 1)
        self.assertAlmostEqual(lwc.lwp, 60.719, 3)
        self.assertAlmostEqual(self.lwc.lwp, 400., 1)

    def test_get_liquid_water_path2(self):
        self.lwc.init_cloud_layers(0, 50)
        lwc = LowWaterCloud(2000., 255., 400., 0)
        lwc.init_cloud_layers(0, 10)
        lwc.get_liquid_water_path()
        self.lwc.get_liquid_water_path()
        self.assertAlmostEqual(lwc.lwp, self.lwc.lwp, 1)

    def test_init_cloud_layers(self):
        self.lwc.init_cloud_layers(0, 100)
        self.lwc.plot_lowcloud('lwc', 'Liquid water content in [g m-3]',
                               '/tmp/test_lowwatercloud_lwc.png')
        self.assertAlmostEqual(len(self.lwc.layers), 22)
        self.assertAlmostEqual(self.lwc.layers[0].z, 0)
        self.assertAlmostEqual(self.lwc.layers[1].z, 50)
        self.assertAlmostEqual(self.lwc.layers[20].press, 800, 0)
        self.assertAlmostEqual(self.lwc.layers[21].z, 2000)

    def test_cloud_layer(self):
        lwc = LowWaterCloud(2000., 255., 400., 0)
        cl = CloudLayer(0, 100, lwc)
        cl1 = CloudLayer(1945, 1955, lwc)
        cl2 = CloudLayer(1970, 1980, lwc)
        cl3 = CloudLayer(1950, 2050, lwc)
        self.assertAlmostEqual(cl.z, 50., 2)
        self.assertAlmostEqual(cl.temp, -5.47, 2)
        self.assertAlmostEqual(cl.press, 1007.26, 2)
        self.assertAlmostEqual(cl.psv, 4.07, 2)
        self.assertAlmostEqual(cl1.lwc, 0.607, 3)
        self.assertAlmostEqual(cl2.lwc, 0.304, 3)
        self.assertAlmostEqual(cl3.lwc, 0., 3)

    def test_cloud_layer_small(self):
        lwc = LowWaterCloud(1000., 235., 400., 950)
        cl = CloudLayer(950, 960, lwc, False)
        cl1 = CloudLayer(960, 970, lwc, False)
        cl2 = CloudLayer(970, 980, lwc, False)
        cl3 = CloudLayer(980, 990, lwc, False)
        cl4 = CloudLayer(990, 1000, lwc, False)
        self.assertAlmostEqual(cl.lwc, 8, 5)
        self.assertAlmostEqual(cl1.lwc, 6, 5)
        self.assertAlmostEqual(cl2.lwc, 4, 5)
        self.assertAlmostEqual(cl3.lwc, 3, 5)
        self.assertAlmostEqual(cl4.lwc, 1, 5)
        self.assertAlmostEqual(lwc.upthres, 49)
        self.assertAlmostEqual(lwc.maxlwc, 8.2, 5)

    def test_cloud_layer_small2(self):
        lwc = LowWaterCloud(1000., 235., 400., 950)
        cl1 = CloudLayer(970, 980, lwc, False)
        cl2 = CloudLayer(980, 990, lwc, False)
        cl3 = CloudLayer(990, 1000, lwc, False)
        self.assertAlmostEqual(cl1.lwc, 4, 5)
        self.assertAlmostEqual(cl2.lwc, 3, 5)
        self.assertAlmostEqual(cl3.lwc, 1, 5)
        self.assertAlmostEqual(lwc.upthres, 49)
        self.assertAlmostEqual(lwc.maxlwc, 8.2, 5)

    def test_get_moist_air_density(self):
        self.lwc.cbh = 0
        empiric_hrho_0 = self.lwc.get_moist_air_density(100000, 0, 273.15,
                                                        True)
        empiric_hrho_20 = self.lwc.get_moist_air_density(101325, 0, 293.15,
                                                         True)
        empiric_humid_hrho_20 = self.lwc.get_moist_air_density(101325,  2338,
                                                               293.15, True)
        empiric_humid_hrho_neg20 = self.lwc.get_moist_air_density(101325,
                                                                  996.3,
                                                                  253.15, True)

        ideal_hrho_15 = self.lwc.get_moist_air_density(101325, 0, 288.15)
        ideal_hrho_20 = self.lwc.get_moist_air_density(101325, 0, 293.15)
        humid_ideal_hrho_20 = self.lwc.get_moist_air_density(101325,  2338,
                                                             293.15)
        humid_ideal_hrho_neg20 = self.lwc.get_moist_air_density(101325,  996.3,
                                                                253.15)

        self.assertAlmostEqual(empiric_hrho_0, 1.276, 3)
        self.assertAlmostEqual(empiric_hrho_20, 1.205, 4)
        self.assertAlmostEqual(ideal_hrho_15, 1.2250, 4)
        self.assertAlmostEqual(ideal_hrho_20, 1.2041, 4)
        self.assertAlmostEqual(humid_ideal_hrho_20, 1.194, 3)
        self.assertAlmostEqual(empiric_humid_hrho_20, 1.1945, 4)
        self.assertAlmostEqual(humid_ideal_hrho_neg20, 1.3892, 4)
        self.assertAlmostEqual(empiric_humid_hrho_neg20, 1.3902, 4)

    def test_get_incloud_mixing_ratio(self):
        self.lwc.cbh = 100
        self.lwc.cth = 1000
        beta_0 = self.lwc.get_incloud_mixing_ratio(500, 1000, 100)
        beta_1 = self.lwc.get_incloud_mixing_ratio(100, 1000, 100)
        beta_2 = self.lwc.get_incloud_mixing_ratio(130, 1000, 100)
        beta_3 = self.lwc.get_incloud_mixing_ratio(175, 1000, 100)
        beta_4 = self.lwc.get_incloud_mixing_ratio(950, 1000, 100)
        self.assertAlmostEqual(beta_0, 0.3)
        self.assertAlmostEqual(beta_1, 0)
        self.assertAlmostEqual(beta_2, 0.12)
        self.assertAlmostEqual(beta_3, 0.3, 2)
        self.assertAlmostEqual(beta_4, 0.3)

    def test_get_incloud_mixing_ratio_limit(self):
        self.lwc.cbh = 950
        self.lwc.cth = 1000
        beta_0 = self.lwc.get_incloud_mixing_ratio(950, 1000, 950)
        beta_1 = self.lwc.get_incloud_mixing_ratio(960, 1000, 950)
        beta_2 = self.lwc.get_incloud_mixing_ratio(970, 1000, 950)
        beta_3 = self.lwc.get_incloud_mixing_ratio(980, 1000, 950)
        beta_4 = self.lwc.get_incloud_mixing_ratio(990, 1000, 950)
        beta_5 = self.lwc.get_incloud_mixing_ratio(1000, 1000, 950)
        self.assertAlmostEqual(beta_0, 0.3)
        self.assertAlmostEqual(beta_1, 0.3)
        self.assertAlmostEqual(beta_2, 0.3)
        self.assertAlmostEqual(beta_3, 0.3, 2)
        self.assertAlmostEqual(beta_4, 0.3)
        self.assertAlmostEqual(beta_5, 0.3)

    def test_optimize_cbh_brute(self):
        self.lwc.thickness = 100
        ret_brute = self.lwc.optimize_cbh(100., method='brute')
        self.assertAlmostEqual(ret_brute, 421., 1)

    def test_optimize_cbh_basin(self):
        self.lwc.thickness = 100
        ret_basin = self.lwc.optimize_cbh(100., method='basin')
        self.assertIn(round(ret_basin, 0), [421, 479, 478, 477])

    def test_optimize_cbh_start(self):
        self.lwc.thickness = 100.
        listresult = []
        listresult.append(self.lwc.optimize_cbh(1000., method='basin'))
        listresult.append(self.lwc.optimize_cbh(900., method='basin'))
        listresult.append(self.lwc.optimize_cbh(800., method='basin'))
        listresult.append(self.lwc.optimize_cbh(700., method='basin'))
        listresult.append(self.lwc.optimize_cbh(600., method='basin'))
        listresult.append(self.lwc.optimize_cbh(500., method='basin'))
        listresult.append(self.lwc.optimize_cbh(400., method='basin'))
        listresult.append(self.lwc.optimize_cbh(300., method='basin'))
        listresult.append(self.lwc.optimize_cbh(200., method='basin'))
        listresult.append(self.lwc.optimize_cbh(100., method='basin'))
        listresult.append(self.lwc.optimize_cbh(0., method='basin'))
        listresult.append(self.lwc.optimize_cbh(-100., method='basin'))
        test = [round(i, 0) == 421 for i in listresult]
        self.assertGreaterEqual(sum(test), 8)

    def test_optimize_cbh_start_thin(self):
        self.thinlwc.thickness = 10.
        listresult = []
        listresult.append(self.thinlwc.optimize_cbh(1000., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(900., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(800., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(700., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(600., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(500., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(400., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(300., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(200., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(100., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(0., method='basin'))
        listresult.append(self.thinlwc.optimize_cbh(-100., method='basin'))
        test = [round(i, 0) == 421 for i in listresult]
        self.assertGreaterEqual(sum(test), 8)

    def test_optimize_cbh_basin_nan(self):
        self.nanlwc.thickness = 100
        ret_basin = self.nanlwc.optimize_cbh(100., method='basin')
        self.assertTrue(np.isnan(ret_basin))

    def test_optimize_cbh_basin_nodata(self):
        self.nodatalwc.thickness = 100
        ret_basin = self.nodatalwc.optimize_cbh(100., method='basin')
        self.assertTrue(np.isnan(ret_basin))

    def test_get_visibility(self):
        lwc = LowWaterCloud(2000., 255., 400., 0, 10e-6)
        vis = self.lwc.get_visibility(1)
        vis2 = self.lwc.get_visibility(1/1000.)
        self.assertAlmostEqual(vis, 3.912, 3)
        self.assertAlmostEqual(vis2, 3912, 0)

    def test_get_liquid_density(self):
        lwc = LowWaterCloud(2000., 285., 400., 0)
        rho1 = self.lwc.get_liquid_density(20, 100e5)
        rho2 = self.lwc.get_liquid_density(4, 1e5)
        rho3 = self.lwc.get_liquid_density(0, 1e5)
        self.assertAlmostEqual(rho1, 1002.66, 3)
        self.assertAlmostEqual(rho2, 999.448, 3)
        self.assertAlmostEqual(rho3, 999.80, 3)

    def test_get_effective_radius(self):
        lwc = LowWaterCloud(1000., 255., 400., reff=10e-6, cbh=0)
        reff_b = lwc.get_effective_radius(0)
        reff_m = lwc.get_effective_radius(500)
        reff_t = lwc.get_effective_radius(lwc.cth)
        self.assertAlmostEqual(reff_b, 1e-6)
        self.assertAlmostEqual(reff_m, 5.5e-6)
        self.assertAlmostEqual(reff_t, 10e-6)

    def test_get_effective_radius_with_cbh(self):
        lwc = LowWaterCloud(1000., 255., 400., reff=10e-6, cbh=100)
        reff_b = lwc.get_effective_radius(100)
        reff_m = lwc.get_effective_radius(550)
        reff_t = lwc.get_effective_radius(lwc.cth)
        self.assertAlmostEqual(reff_b, 1e-6)
        self.assertAlmostEqual(reff_m, 5.5e-6)
        self.assertAlmostEqual(reff_t, 10e-6)

    def test_get_fog_cloud_height(self):
        lwc = LowWaterCloud(2000., 275., 400., 100, 10e-6)
        lwc.init_cloud_layers(100, 50)
        fbh = lwc.get_fog_base_height()
        self.assertAlmostEqual(fbh, 125, 0)

    def test_get_fog_cloud_height2(self):
        lwc = LowWaterCloud(1000., 275., 100., 100., 10e-6)
        lwc.init_cloud_layers(100, 10)
        lwp = lwc.get_liquid_water_path()
        cbh = lwc.optimize_cbh(lwc.cbh)
        fbh = lwc.get_fog_base_height()
        self.assertAlmostEqual(lwc.lwp, 100, 3)
        self.assertAlmostEqual(lwc.maxlwc, 0.494, 3)
        self.assertAlmostEqual(fbh, 612, 0)


def suite():
    """The test suite for test_lowwatercloud.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_LowWaterCloud))

    return mysuite


if __name__ == "__main__":
    unittest.main()
