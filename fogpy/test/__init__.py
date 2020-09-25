#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Fogpy developers

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

"""The fogpy test suite.
"""

from fogpy.test import (test_lowwatercloud,
                        test_filters,
                        test_algorithms
                        )

import unittest


def suite():
    """The global test suite.
    """

    mysuite = unittest.TestSuite()
    mysuite.addTests(test_lowwatercloud.suite())
    mysuite.addTests(test_filters.suite())
    mysuite.addTests(test_algorithms.suite())

    return mysuite
