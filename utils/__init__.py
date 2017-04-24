#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017
# Author(s):
#   Thomas Leppelt <thomas.leppelt@dwd.de>

# This file is part of the fogpy package.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""PP Package initializer.
"""

import datetime
import itertools
import os

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])


def ncycle(iterable, n):
    for item in itertools.cycle(iterable):
        for i in range(n):
            yield item


def get_time_period(start, end, step):
    """Create time series from given start to end by certain interval

    Keyword arguments:
        start    Start time as string in %Y%m%%%d%H%M format
        end    End time as string in %Y%m%%%d%H%M format
    """
    # Define time series for analysis
    dt = datetime.datetime.strptime(start, "%Y%m%d%H%M")
    tend = datetime.datetime.strptime(end, "%Y%m%d%H%M")

    ts = []
    if isinstance(step, list):
        c = ncycle(step, 1)
    else:
        c = ncycle([step], 1)

    while dt < tend:
        ts.append(dt)
        ti = c.next()
        tstep = datetime.timedelta(minutes=ti)
        dt += tstep

    return(ts)
