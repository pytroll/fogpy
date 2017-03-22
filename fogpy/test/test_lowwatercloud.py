#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017
# Author(s):
#   Thomas Leppelt <thomas.leppelt@dwd.de>
""" This module test the low cloud water class """

import unittest
from lowwatercloud import LowWaterCloud
from lowwatercloud import CloudLayer


class Test(unittest.TestCase):

    def setUp(self):
        self.lwc = LowWaterCloud(2000., 255., 400.)

    def tearDown(self):
        pass

    def test_get_sat_vapour_pressure_buck(self):
        psv_m50 = self.lwc.get_sat_vapour_pressure(-50)
        psv_m20 = self.lwc.get_sat_vapour_pressure(-20)
        psv_0 = self.lwc.get_sat_vapour_pressure(0)
        psv_20 = self.lwc.get_sat_vapour_pressure(20)
        psv_50 = self.lwc.get_sat_vapour_pressure(50)
        self.assertAlmostEqual(round(psv_m50, 3), 0.064)
        self.assertAlmostEqual(round(psv_m20, 3), 1.256)
        self.assertAlmostEqual(round(psv_0, 3), 6.112)
        self.assertAlmostEqual(round(psv_20, 3), 23.383)
        self.assertAlmostEqual(round(psv_50, 3), 123.494)

    def test_get_sat_vapour_pressure_magnus(self):
        psv_m50 = self.lwc.get_sat_vapour_pressure(-50, 'magnus')
        psv_m20 = self.lwc.get_sat_vapour_pressure(-20, 'magnus')
        psv_0 = self.lwc.get_sat_vapour_pressure(0, 'magnus')
        psv_20 = self.lwc.get_sat_vapour_pressure(20, 'magnus')
        psv_50 = self.lwc.get_sat_vapour_pressure(50, 'magnus')
        self.assertAlmostEqual(round(psv_m50, 3), 0.064)
        self.assertAlmostEqual(round(psv_m20, 3), 1.254)
        self.assertAlmostEqual(round(psv_0, 3), 6.108)
        self.assertAlmostEqual(round(psv_20, 3), 23.420)
        self.assertAlmostEqual(round(psv_50, 3), 123.335)

    def test_get_air_pressure(self):
        pa_1 = self.lwc.get_air_pressure(610)
        pa_2 = self.lwc.get_air_pressure(0)
        self.assertAlmostEqual(round(pa_1, 2), 942.08)
        self.assertAlmostEqual(round(pa_2, 2), 1013.25)

    def test_get_moist_adiabatic_lapse_temp(self):
        temp_0 = self.lwc.get_moist_adiabatic_lapse_temp(500, 1500, 0)
        self.assertAlmostEqual(round(temp_0, 1), 6.5)

    def test_get_vapour_mixing_ratio(self):
        vmr = self.lwc.get_vapour_mixing_ratio(1007.26, 8.44)
        self.assertAlmostEqual(round(vmr, 2), 5.26)

    def test_get_cloud_based_vapour_mixing_ratio(self):
        self.lwc.cbh = 0
        cb_vmr = self.lwc.get_cloud_based_vapour_mixing_ratio()
        self.assertAlmostEqual(round(cb_vmr, 2), 2.57)

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
        self.assertAlmostEqual(round(lwc, 3), 1.519)

    def test_get_liquid_water_path(self):
        self.lwc.init_cloud_layers(421., 100)
        self.lwc.get_liquid_water_path()
        lwc = LowWaterCloud(2000., 255., 400., 0)
        cl = CloudLayer(1900, 2000, lwc)
        lwc.get_liquid_water_path()
        self.assertAlmostEqual(len(lwc.layers), 1)
        self.assertAlmostEqual(round(lwc.lwp, 3), 60.719)
        self.assertAlmostEqual(round(self.lwc.lwp, 1), 400.)

    def test_get_liquid_water_path2(self):
        self.lwc.init_cloud_layers(0, 50)
        lwc = LowWaterCloud(2000., 255., 400., 0)
        lwc.init_cloud_layers(0, 10)
        lwc.get_liquid_water_path()
        self.lwc.get_liquid_water_path()
        self.assertAlmostEqual(round(lwc.lwp, 1), round(self.lwc.lwp, 1))

    def test_init_cloud_layers(self):
        self.lwc.init_cloud_layers(0, 100)
        self.lwc.plot_lowcloud('lwc', 'Liquid water content in [g m-3]',
                               '/tmp/test_lowwatercloud_lwc.png')
        self.assertAlmostEqual(len(self.lwc.layers), 22)
        self.assertAlmostEqual(self.lwc.layers[0].z, 0)
        self.assertAlmostEqual(self.lwc.layers[1].z, 50)
        self.assertAlmostEqual(round(self.lwc.layers[20].press, 0), 800)
        self.assertAlmostEqual(self.lwc.layers[21].z, 2000)

    def test_cloud_layer(self):
        lwc = LowWaterCloud(2000., 255., 400., 0)
        cl = CloudLayer(0, 100, lwc)
        cl1 = CloudLayer(1945, 1955, lwc)
        cl2 = CloudLayer(1970, 1980, lwc)
        cl3 = CloudLayer(1950, 2050, lwc)
        self.assertAlmostEqual(round(cl.z, 2), 50.)
        self.assertAlmostEqual(round(cl.temp, 2), -5.47)
        self.assertAlmostEqual(round(cl.press, 2), 1007.26)
        self.assertAlmostEqual(round(cl.psv, 2), 4.07)
        self.assertAlmostEqual(round(cl1.lwc, 3), 0.607)
        self.assertAlmostEqual(round(cl2.lwc, 3), 0.304)
        self.assertAlmostEqual(round(cl3.lwc, 3), 0.)

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

        self.assertAlmostEqual(round(empiric_hrho_0, 3), 1.276)
        self.assertAlmostEqual(round(empiric_hrho_20, 4), 1.205)
        self.assertAlmostEqual(round(ideal_hrho_15, 4), 1.2250)
        self.assertAlmostEqual(round(ideal_hrho_20, 4), 1.2041)
        self.assertAlmostEqual(round(humid_ideal_hrho_20, 3), 1.194)
        self.assertAlmostEqual(round(empiric_humid_hrho_20, 4), 1.1945)
        self.assertAlmostEqual(round(humid_ideal_hrho_neg20, 4), 1.3892)
        self.assertAlmostEqual(round(empiric_humid_hrho_neg20, 4), 1.3902)

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
        self.assertAlmostEqual(round(beta_3, 2), 0.3)
        self.assertAlmostEqual(beta_4, 0.3)

    def test_optimize_cbh(self):
        self.lwc.thickness = 100
        ret_brute = self.lwc.optimize_cbh(100., method='brute')
        ret_basin = self.lwc.optimize_cbh(100., method='basin')
        self.assertAlmostEqual(round(ret_brute, 1), 421.)
        self.assertAlmostEqual(round(ret_basin, 1), 379.1)

if __name__ == "__main__":
    unittest.main()
