#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020
# Author(s):
#   Thomas Leppelt <thomas.leppelt@dwd.de>, Gerrit Holl <gerrit.holl@dwd.de>

# This file is part of the fogpy package.

# fogpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# fogpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with fogpy.  If not, see <http://www.gnu.org/licenses/>.

"""Small utilities needed by Fogpy
"""

import logging

import requests

logger = logging.getLogger(__name__)


def dl_dem(dem):
    """Download Digital Elevation Model

    Download a Digital Elevation Model (DEM) from Zenodo.

    The source URI is derived from the destination path.

    Args:
        dem (pathlib.Path): Destination
    """
    src = "https://zenodo.org/record/3885398/files/" + dem.name

    if dem.exists():
        raise FileExistsError("Already exists: {dem!s}")
    r = requests.get(src)
    logger.info("Downloading {src!s} to {dem!s}")
    dem.parent.mkdir(exist_ok=True, parents=True)
    with dem.open(mode="wb") as fp:
        fp.write(r.content)
