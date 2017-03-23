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

from setuptools import setup, find_packages
import os
import imp

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep))

version = imp.load_source('fogpy.version', 'fogpy/version.py')
#here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='fogpy',
    version=version.__version__,
    url='https://github.com/m4sth0/fogpy',
    license='GNU general public license version 3',
    author='Thomas Leppelt',
    author_email='thomas.leppelt@gmail.com',
    description='Satellite based fog and low stratus detection and nowcasting',
    packages=['fogpy'],
    include_package_data=True,
    platforms='any',
    test_suite='fogpy.test.suite',
    classifiers = [
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
    install_requires=['numpy >=1.4.1',
                    'scipy >=0.17.0',
                    'matplotlib >=1.4.2',
                    'mpop >=v1.3.1',
                    'pyorbital >= v0.2.3'
                    ],
    tests_require=[],
    #extras_require={
    #    'testing': ['pytest'],
    #}
)