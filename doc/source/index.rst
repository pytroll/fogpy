.. fogpy documentation master file, created by
   sphinx-quickstart on Thu Apr 20 12:31:41 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================
Welcome to fogpy's documentation!
=================================

.. image:: ./fogpy_logo.png

This package provide algorithmns and methods for satellite based detection and
nowcasting of fog and low stratus clouds (FLS). 

Related FogPy Version: 1.1.3

It utilizes several functionalities from the pytroll_ project for weather
satellite data processing in Python. The remote sensing algorithmns are
currently implemented for the geostationary Meteosat Second Generation (MSG)
satellites. But it is designed to be easly extendable to support other
meteorological satellites in future.

Contents:

.. _pytroll: http://pytroll.org/
.. toctree::
   :maxdepth: 2
   
   install
   quickstart
   algorithms
   filters
   lowcloud



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

