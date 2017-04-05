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

""" This module implements a class for a 1D low water cloud model.
The approach can be used to determine fog cloud base heights by known
cloud top height and temperature and cloud liquid water path, e.g. from
satellite retrievals.
The implemented approch is based on a publication:
--- Detecting ground fog from space – a microphysics-based approach
Jan Cermak and Joerg Bendi, 2010 ---
"""
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import basinhopping
from scipy.optimize import brute

# Configure logger.
logger = logging.getLogger('lowwatercloud')
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - '
                           '%(message)s', level=logging.INFO)


class CloudLayer(object):
    """ This class represent a cloud layer - 1D representation of a
    cloud section from its vertical profile with defined extent and homogenius
    cloud parameters.
    The layer is defined by the bottom and top height in the cloud profile"""
    def __init__(self, bottom, top, lowcloud, add=True):
        self.bottom = bottom  # Bottom height of the cloud layer
        self.top = top  # Top height of the cloud layer
        self.z = top - (top - bottom) / 2  # Central layer height

        # Get temperaure and air presure
        self.temp = lowcloud.get_moist_adiabatic_lapse_temp(self.z,
                                                            lowcloud.cth,
                                                            lowcloud.ctt,
                                                            True)
        self.press = lowcloud.get_air_pressure(self.z)

        # Calculate Vapour pressure and mixing ratio
        self.psv = lowcloud.get_sat_vapour_pressure(self.temp,
                                                    lowcloud.vapour_method)
        self.vmr = lowcloud.get_vapour_mixing_ratio(self.press, self.psv)

        # Get liquid water mixing ratio
        if lowcloud.cb_vmr is None:
            lowcloud.get_cloud_based_vapour_mixing_ratio()

        self.lmr = lowcloud.get_liquid_mixing_ratio(lowcloud.cb_vmr, self.vmr)

        # Get layer density
        self.rho = lowcloud.get_moist_air_density(self.press * 100,
                                                  self.psv * 100,
                                                  self.check_temp(self.temp,
                                                                  'kelvin'))

        # Get layer liquid water density
        self.lrho = lowcloud.get_liquid_density(self.press * 100,
                                                self.check_temp(self.temp,
                                                                'celsius'))

        # Get in cloud mixing ratio beta
        self.beta = lowcloud.get_incloud_mixing_ratio(self.z, lowcloud.cth,
                                                      lowcloud.cbh)

        # Get liquid water content
        self.lwc = lowcloud.get_liquid_water_content(self.z, lowcloud.cth,
                                                     self.rho, self.lmr,
                                                     self.beta,
                                                     lowcloud.upthres,
                                                     lowcloud.maxlwc)

        # Get layer effective radius
        self.reff = lowcloud.get_effective_radius(self.z)

        # Get layer extinction coefficient
        self.extinct = lowcloud.get_extinct(self.lwc, self.reff,
                                            (self.lrho * 1000.))

        # Get visibility
        self.visibility = lowcloud.get_visibility(self.extinct)

        # Add cloud layer to low water cloud object
        if add:
            lowcloud.layers.append(self)

        logger.debug('New cloud layer: z: {} m | t: {} °C | p: {} hPa | '
                     'psv: {} hPa | vmr: {} g kg-1 | lmr: {} g kg-1 | '
                     'beta: {} | rho: {} kg m-3 | lwc: {} g m-3'
                     .format(self.z, self.temp, round(self.press, 3),
                             round(self.psv, 3), round(self.vmr, 3),
                             round(self.lmr, 3), round(self.beta, 3),
                             round(self.rho, 3), round(self.lwc, 3)))

    @classmethod
    def check_temp(self, temp, unit='celsius'):
        """ Check for plausible range of temperature value for given unit.
        Convert if required"""
        if unit is 'celsius':
            if temp > 60:
                result = temp - 273.15
                logger.debug('Temperature {} is in Kelvin. Auto converting to'
                             ' {} °C'.format(temp, result))
            else:
                result = temp
        elif unit is 'kelvin':
            if temp <= 60 or temp < 0:
                result = temp + 273.15
                logger.debug('Temperature {} is in Celsius. Auto converting to'
                             ' {} K'.format(temp, result))
            else:
                result = temp

        return(result)


class LowWaterCloud(object):
    """ A class to simulate the water content of a low cloud and calculate its
    meteorological properties"""
    def __init__(self, cth=None, ctt=None, cwp=None, cbh=0, reff=None,
                 cbt=None, upthres=50., lowthres=75., thickness=10):
        self.cth = cth  # Cloud top height
        self.ctt = ctt  # Cloud top temperature
        self.cbh = cbh  # Cloud base height
        self.cbt = cbt   # Cloud base temperature
        self.cwp = cwp  # Cloud water path
        self.reff = reff  # Droplet effective radius
        self.layers = []  # List of cloud layers
        self.vapour_method = "magnus"  # Method for saturated vapour pressure
        self.cb_vmr = None
        self.upthres = upthres
        self.lowthres = lowthres
        self.thickness = thickness
        # Get maximal liquid water content underneath cloud top
        self.maxlwc = None

    def init_cloud_layers(self, init_cbh, thickness, overwrite=True):
        """Method to initialize cloud layers and corresponding parameters.
        the method needs a initial cloud base height and thickness in [m]"""
        self.cbh = init_cbh
        self.get_cloud_based_vapour_mixing_ratio()

        # Get maximal liquid water content underneath cloud top
        maxlwc_layer = CloudLayer(self.cth - self.upthres - 5,
                                  self.cth - self.upthres + 5,
                                  self, False)
        self.maxlwc = maxlwc_layer.lwc

        # Calculate layer properties
        # Contitional resetting cloud layers
        if overwrite:
            self.layers = []
        # Cloud base layer
        CloudLayer(init_cbh - 5, init_cbh + 5, self)
        # Loop over layers
        layerrange = np.arange(init_cbh, self.cth, thickness)
        for b in layerrange:
            CloudLayer(b, b + thickness, self)
        # Cloud top layer
        CloudLayer(self.cth - 5, self.cth + 5, self)

        #logger.info("Initialize {} cloud layers with {} m thickness"
        #            " and {} m cbh"
        #            .format(len(self.layers), thickness, init_cbh))

    def get_cloud_base_height(self, start=0, method='basin'):
        """ Calculate cloud base height [m]"""
        # Calculate cloud base height
        self.cbh = self.optimize_cbh(start, method='basin')

        return self.cbh

    def get_fog_base_height(self):
        """This method calculate the fog cloud base height for low clouds
        with visibilities below 1000 m
        """
        print(self.reff)
        print([l.z for l in self.layers])
        print([l.lwc for l in self.layers])
        print([l.reff for l in self.layers])
        print([l.extinct for l in self.layers])
        print([l.visibility for l in self.layers])
        fog_z = [l.z for l in self.layers if (l.visibility <= 1000) & (l.visibility is not None)]
        print(fog_z)
        fbh = min(fog_z)  # Get lowest heights with visibility treshold
        print(fbh)
        return fbh

    def get_liquid_water_content(self, z, cth, hrho, lmr, beta, thres,
                                 maxlwc=None):
        """ Calculate liquid water content [g m-3] by air density and
        liquid water mixing ratio"""
        if z > cth - thres:
            if maxlwc is None:
                maxlwc_layer = CloudLayer(self.cth - self.upthres - 5,
                                          self.cth - self.upthres + 5,
                                          self, False)
                maxlwc = maxlwc_layer.lwc
                self.maxlwc = maxlwc

            lwc = (cth - z) / (thres) * maxlwc
        else:
            lwc = (1 - beta) * hrho * lmr

        return lwc

    @classmethod
    def get_moist_air_density(self, pa, pv, temp, empiric=False):
        """ Calculate air density for humid air with known pressure and water
        vapour pressure and temperature"""
        molar_d = 0.028964  # kg mol-1
        molar_w = 0.018016  # kg mol-1
        Rv = 461.495  # Specific gas constant for water vapour J kg-1 K-1
        Rd = 287.058  # Specific gas constant for dry air J kg-1 K-1
        R = 8.31432  # Universal gas constant [N m mol-1 K-1]
        d_sea = 1.2929  # Density of dry air at sea level
        torr_pa = 0.00750063755  # Factor to convert between Pa and Torr
        if temp <= 60:
            newtemp = temp + 273.15
            logger.debug('Temperature {} is in Celsius. Auto converting to'
                         ' {} K'.format(temp, newtemp))
            temp = newtemp
        if empiric:
            # hrho = d_sea * (273.15 / temp) * ((pa * torr_pa - 0.3783 * pv *
            #                                    torr_pa) / 760.)
            hrho = d_sea * (273.15 / temp) * ((pa - 0.3783 * pv) / (1.013 *
                                              10**5))
        else:
            hrho = ((pa - pv) / (Rd * temp)) + (pv / (Rv * temp))

        return hrho

    @classmethod
    def get_moist_adiabatic_lapse_temp(self, z, cth, ctt, convert=False):
        """ Calculate air temperature for height z [K] following a moist
        adiabatic lapse rate.
        Requires values for cloud top height and temperature
        e.g. known from satellite retrievals"""
        malr = 0.0065  # ##  MALR: Moist adiabatic lapse rate [K m-1]
        temp = (cth - z) * malr + ctt

        # Optional convertion to Celsius degrees.
        if convert:
            temp = temp - 273.15

        return temp

    @classmethod
    def get_sat_vapour_pressure(self, temp, mode='buck', convert=False):
        """ Calculate satured water vapour pressure for temperature [hPa]
        using different empirical approaches.
        Options: Buck, Magnus

        Convert temperatures in K to °C"""
        if convert:
            newtemp = temp - 273.15
            logger.info("Converting temperature {} K to {} °C".format(temp,
                                                                      newtemp))
            temp = newtemp
        elif temp > 60:
            newtemp = temp - 273.15
            logger.debug('Temperature {} is in Kelvin. Auto converting to'
                         ' {} °C'.format(temp, newtemp))
            temp = newtemp

        if mode == 'buck':
            psv = 0.61121 * math.exp((18.678 - temp / 234.5) *
                                     (temp / (257.14 + temp)))
        elif mode == 'magnus':
            const1 = 6.1078
            if temp > 0:
                const2 = 17.08085
                const3 = 234.175
            else:
                const2 = 17.84362
                const3 = 245.425
            psv = const1 * math.exp(const2 * temp / (const3 + temp))
        # Convert to hPa
        if mode == 'buck':
            result = psv * 10
        else:
            result = psv

        return result

    @classmethod
    def get_vapour_pressure(self, z, temp):
        """ Calculate water vapour pressure for height z [hPa] """
        # TODO Finish implementation
        wdensity = 0  # Density of water vapour
        gconst = 461.51  # Gas constante of water vapour  in [J kg-1 K-1]
        # Calculate water vapur pressure
        vp = wdensity * gconst * temp

        return vp

    @classmethod
    def get_air_pressure(self, z, elevation=0):
        """ Calculate ambient air pressure for height z [hPa] """
        p_sea = 101325.  # Static pressure (pressure at sea level) [Pa]
        t_sea = 288.15  # Standard temperature (temperature at sea level) [K]
        TL_rate = 0.0065  # Standard temperature lapse rate [K/m]
        R = 8.31432  # Universal gas constant [N m mol-1 K-1]
        G = 9.80665  # Gravitational acceleration constant [m s-2]
        M = 0.0289644  # Molar mass of Earth’s air [kg/mol]

        power = -5.255877
        fraction = t_sea / (t_sea + (TL_rate * (z - elevation)))
        pa = 100 * ((44331.514 - z) / 11880.516) ** (1 / 0.1902632)

        return pa / 100

    @classmethod
    def get_vapour_mixing_ratio(self, pa, pv):
        """ Calculate water vapour mixing ratio for given ambient pressure and
        water vapur pressure. Also usabale under saturated conditions"""
        # Calculate water vapour mixing ratio
        vmr = 621.97 * pv / (pa - pv)

        return vmr

    @classmethod
    def get_liquid_mixing_ratio(self, cb_vmr, vmr):
        """ Calculate liquid water mixing ratio for given water vapour mixing
        ratio in a certain height and the maximum water vapour mixing ratio at
         cloud base condensation level [g/kg] """
        lmr = cb_vmr - vmr

        return lmr

    def get_cloud_based_vapour_mixing_ratio(self):
        # Get temperaure and air presure
        temp = self.get_moist_adiabatic_lapse_temp(self.cbh, self.cth,
                                                   self.ctt, True)
        press = self.get_air_pressure(self.cbh)

        # Calculate Vapour pressure and mixing ratio
        psv = self.get_sat_vapour_pressure(temp, self.vapour_method)
        cb_vmr = self.get_vapour_mixing_ratio(press, psv)
        self.cb_vmr = cb_vmr

        return cb_vmr

    @classmethod
    def get_incloud_mixing_ratio(self, z, cth, cbh, lowthres=75., upthres=50.):
        """ Calculate in-cloud mixing ratio for given cloud height parameter"""
        midbeta = 0.3 * cth / 1000  # Fixed in cloud mixing ratio
        # Separation in three major cloud layers
        # Apply fixed value for middle layer
        if z > cbh + lowthres and z < cth - upthres:
            beta = midbeta
        # Apply zero value for upper layer
        elif z >= cth - upthres:
            beta = midbeta
        # Apply linear increase from zero to fixed value in the lower layer
        elif z <= (cbh + lowthres):
            beta = (z - cbh) / (lowthres) * midbeta

        return beta

    def get_liquid_water_path(self):
        """ Calculate liquid water path for given cloud layers [g m-2] """
        z = np.array([l.top - l.bottom for l in self.layers])
        lwc = np.array([l.lwc for l in self.layers])
        # Get sum of single layer water path
        lwp = np.sum(z * lwc)

        self.lwp = lwp

        return lwp

    def optimize_cbh(self, start, method='basin'):
        """ Find best fitting cloud base height by comparing calculated
        liquid water path with given satellite retrieval.
        Minimization with basinhopping or brute force algorithm
        from python scipy package"""
        if method == 'basin':
            minimizer_kwargs = {"method": "BFGS", "bounds": (0, self.cth)}
            ret = basinhopping(self.minimize_cbh,
                               start,
                               T=10.0,
                               minimizer_kwargs=minimizer_kwargs,
                               niter=100,
                               niter_success=5)
            result = ret.x[0]
            logger.info('Optimized liquid water path: start cbh: {}, cth: {},'
                        ' observed lwp {} --> result lwp: {},'
                        ' calibrated cbh: {}'
                        .format(start, self.cth, self.cwp, self.lwp, ret.x[0]))
        elif method == 'brute':
            ranges = slice(0, self.cth, 1)
            ret = brute(self.minimize_cbh, (ranges,), finish=None)
            result = ret
        # Set class variable for cloud base height
        self.cbh = result

        return result

    def minimize_cbh(self, x):
        """Minimization function for liquid water path"""
        x = np.reshape(x, (1,))
        self.init_cloud_layers(x[0], self.thickness, True)
        lwp = self.get_liquid_water_path()
        diff = abs(lwp - self.cwp)

        return diff

    def get_liquid_density(self, temp, press):
        """Calculate the liquid water density in [kg m-3]
        """
        t0 = 0  # Reference temperature in [°C]
        rho_0 = 999.8  # Density of water for 0°C:  [kg/m3]
        p0 = 1e5  # Air pressure at 0°C in [Pa]
        beta = 0.000088  # Expansion coefficient of water at 10oC: [m3/m3°C]
        E = 2.15e9  # Bulk modulus of water:  [N/m2]
        rho = rho_0 / (1 + beta * (temp - t0)) / (1 - (press - p0) / E)

        return rho

    def get_visibility(self, extinct, contrast=0.02):
        """Calculate visibility in [m] for given cloud layer.
        Extinction is directly related to visibility by
        Koschmieder’s law:
        """
        if extinct is None:
            return None
        else:
            vis = (1 / extinct) * math.log(1 / contrast)
        return vis

    def get_extinct(self, lwc, reff, rho):
        """ Calculate extingtion coeficient [m-1]
        The extinction therefore is a combination of radiation loss by
        (diffuse) scattering and molecular absorption.
        Required are the liquid water content, effective radius and
        liquid water density
        TODO: Recheck the unit of liquid water density g or kg? Should be in g
        """
        if reff is None:
            return None
        elif lwc == 0:
            return None
        else:
            extinct = 3 * lwc / (2 * reff * rho)

        return extinct

    def get_effective_radius(self, z):
        """ The droplet effective radius in [um] for each level is computed on
        the assumptions that reff retrieved at 3.9 μm is the cloud top value,
        Cloud base reff is at 1 μm and the intermediate values are scaled
        linearly in between.
        """
        if self.reff is None:
            return None
        else:
            reff = 1e-6 + ((self.reff - 1e-6) / (self.cth -
                                                 self.cbh)) * (z - self.cbh)

        return reff

    def plot_lowcloud(self, para, xlabel=None, save=None):
        """ Plotting of selected low water cloud parameters"""
        if self.layers == []:
            logger.info("No layer found. Nothing to plot")
        heights = [getattr(l, 'z') for l in self.layers]
        paralist = [getattr(l, para) for l in self.layers]
        plt.plot(paralist, heights)
        plt.ylabel('Cloud height z in [m]')
        if xlabel is not None:
            plt.xlabel(xlabel)
        else:
            plt.xlabel(para)
        if save is not None:
            plt.savefig(save)
        else:
            plt.show()
