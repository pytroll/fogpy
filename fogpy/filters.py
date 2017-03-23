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

"""This module implements an basic algorithm filter class
and several class instances for satellite fog detection applications"""

import numpy as np


class NotApplicableError(Exception):
    """Exception to be raised when a filter is not applicable."""
    pass


class BaseArrayFilter(object):
    """This super filter class provide all functionalities to apply a filter
    funciton on a given numpy array representing a satellite image and return
    the filtered masked array as result"""
    def __init__(self, arr, **kwargs):
        if isinstance(arr, np.ma.MaskedArray):
            self.arr = arr
            self.inmask = arr.mask
        elif isinstance(arr, np.ndarray):
            self.arr = arr
        else:
            raise ImportError('The filter <{}> needs a valid 2d numpy array '
                              'as input'.format(self.__class__.__name__))
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                self.__setattr__(key, value)
            self.result = None
            self.mask = None

    def apply(self):
        """Apply the given filter function"""
        if self.isapplicable():
            self.filter_function()
            self.check()
        else:
            raise NotApplicableError('Array filter <{}> is not applicable'
                                     .format(self.__class__.__name__))
        return(self.result)

    def isapplicable(self):
        """Test filter applicability"""
        ret = True
        return(ret)

    def filter_function(self):
        """Filter routine"""
        self.mask = np.ones(self.arr.shape)
         
        self.result = np.ma.array(self.arr, mask=self.mask)
        
        return(True)

    def check(self):
        """Check filter results for plausible results"""
        ret = True
        return(ret)



