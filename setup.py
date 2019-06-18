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

""" Setup file for fogpy"""

from setuptools import setup
import os
import subprocess

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

# get commits since last tagged version
cp = subprocess.run(
    ["git", "describe", "--tags"],
    stdout=subprocess.PIPE,
    check=True)
so = cp.stdout.strip().decode("utf-8")

# get branch name
cp = subprocess.run(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    stdout=subprocess.PIPE,
    check=True)
br = cp.stdout.strip().decode("utf-8")

# optionally add commit/branch to version string
if "-" in so:  # we're not at a tagged version
    version = so[1:].replace("-", "+dev", 1).replace("-", ".")
    if br != "master":
        version += "." + br
else:
    version = so[1:]

setup(
    name='fogpy',
    version=version,
    url='https://github.com/satpy/fogpy',
    license='GNU general public license version 3',
    author='Thomas Leppelt, Gerrit Holl',
    author_email='thomas.leppelt@gmail.com; gerrit.holl@dwd.de',
    description='Satellite based fog and low stratus detection and nowcasting',
    packages=['fogpy', 'fogpy.utils'],
    include_package_data=True,
    data_files=[(os.path.join('etc'),
                 [os.path.join('etc', 'fog_testdata.npy')])],
    platforms='any',
    test_suite='fogpy.test.suite',
    python_requires=">=3.7",
    classifiers=[
        'Programming Language :: Python',
        'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Environment :: Web Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Physics',
        ],
    install_requires=['numpy >= 1.16',
                      'scipy >= 1.2',
                      'matplotlib >=1.4.2',
                      'pyorbital >= 1.5.0',
                      'trollimage >= 1.8.0',
                      "satpy >= 0.15",
                      'pyresample >= 1.11',
                      "opencv-python >= 4.1",
                      'trollbufr >= 0.10'],
    tests_require=[],
)
