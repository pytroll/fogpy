============
 Fogpy usage in a nutshell
============

The package uses OOP extensively, to allow higher level metaobject handling.

For this tutorial, we will use a MSG scene for creating different 
fog products.

Import satellite data first
===========================

We start with the PyTroll package *mpop*. This package provide all functionalities 
to import and calibrate a MSG scene from HRIT files. Therefore you should make sure 
that mpop is properly configured and all environment variables like *PPP_CONFIG_DIR* 
are set and the HRIT files are in the given search path. For more guidance look up 
in the `mpop`_ documentation

.. _mpop: http://mpop.readthedocs.io/en/latest/install.html#getting-the-files-and-installing-them/

.. note::
	Make sure *mpop* is correctly configured!

Ok, let's get it on::

    >>> from datetime import datetime
    >>> from mpop.satellites import GeostationaryFactory
    >>> time = datetime(2013, 12, 12, 10, 0)
    >>> msg_scene = GeostationaryFactory.create_scene(satname="meteosat",
    >>>                                               satnumber='10',
    >>>                                               instrument="seviri",
    >>>                                               time_slot=time)
    >>> msg_scene.load()

We imported a MSG scene from  12. December 2017 and loaded all twelve channels into the scene object.

Now we want to look at the IR 10.8 channel::

	>>> msg_scene.image[10.8].show()

.. image:: ./fogpy_docu_example_1.png

Everything seems correctly imported. We see a full disk image. So lets see if we can resample it to a central European region::

	>>> eu_scene = msg_scene.project("eurol")
	Computing projection from meteosatseviri[-5567248.07417344 -5570248.47733926  5570248.47733926  5567248.07417344](3712, 3712) to eurol...
	Projecting channel IR_120 (12.000000μm)...
	Projecting channel IR_016 (1.640000μm)...
	Projecting channel IR_087 (8.700000μm)...
	Projecting channel WV_062 (6.250000μm)...
	Projecting channel VIS008 (0.810000μm)...
	Projecting channel WV_073 (7.350000μm)...
	Projecting channel VIS006 (0.635000μm)...
	Projecting channel IR_039 (3.920000μm)...
	Projecting channel IR_097 (9.660000μm)...
	Projecting channel IR_108 (10.800000μm)...
	Projecting channel IR_134 (13.400000μm)...
	Computing projection from meteosatseviri[-5568247.98732816 -5569248.12167703  5569248.12167703  5568247.98732816](11136, 11136) to eurol...
	Projecting channel HRV (0.700000μm)...
	>>> eu_scene.image[10.8].show()

.. image:: ./fogpy_docu_example_2.png

A lot of clouds are present over central Europe. Let's test a fog RGB composite to find some low clouds:: 

	>>> fogimg = eu_scene.image.fog()
	>>> fogimg.show()

.. image:: ./fogpy_docu_example_3.png

The reddish and dark colored clouds represent cold and high altitude clouds, 
whereas the yellow-greenish color over central and eastern Europe is an indication for low clouds and fog.

Continue with more metadata
===========================

In the next step we want to create a fog and low stratus (FLS) composite for the imported scene.
Therefore we have to provide some additional input data like elevation and cloud microphysical parameters.

Starting with the elevation information. Here we use a middle European section 
of the 1x1 km digital elevation model provided by the European Environmental Agency (`EEA`_).

.. _EEA: https://www.eea.europa.eu/data-and-maps/data/copernicus-land-monitoring-service-eu-dem
.. _mipp: https://github.com/pytroll/mipp

.. image:: ./fogpy_docu_example_5.png
	:scale: 74 %

Let's import this DEM data and projection information with the geotiff import 
tool of the `mipp`_ package and create a area definition object for it::

	>>> from mipp import read_geotiff as gtiff
	>>> tiff = "/Path/To/Geotiff/dem_eu_1km.tif"
	>>> params, dem = gtiff.read_geotiff(tiff)
	>>> tiffarea = gtiff.tiff2areadef(params['projection'], params['geotransform'],
                              		  dem.shape)
	GEO transform: (3483900.0, 999.8606811145511, 0.0, 3754425.0, 0.0, -1000.2693282636249)
	Band(1) type: Float32, size 1615 x 1578
	fetched array: <type 'numpy.ndarray'> (1578, 1615) float32 [-200 -> 1099.54 -> 9999]   


Now we have the elevation data as numpy array and the projection information as dictionary. 
The DEM projection differs from the MSG scene one::

	>>> tiffarea
	Area ID: laea_7238
	Description: laea_7238
	Projection ID: laea
	Projection: {'ellps': 'GRS80', 'lat_0': '52', 'lon_0': '10', 'proj': 'laea', 'towgs84': '0,0,0,0,0,0,0', 'units': 'm', 'x_0': '4321000', 'y_0': '3210000'}
	Number of columns: 1615
	Number of rows: 1578
	Area extent: (3483900.0, 2176000.0, 5098675.0, 3754425.0)
	>>> eu_scene.area
	Area ID: eurol
	Description: Euro 3.0km area - Europe
	Projection ID: ps60wgs84
	Projection: {'ellps': 'WGS84', 'lat_0': '90', 'lat_ts': '60', 'lon_0': '0', 'proj': 'stere'}
	Number of columns: 2560
	Number of rows: 2048
	Area extent: (-3780000.0, -7644000.0, 3900000.0, -1500000.0)
	
Lambert Azimuthal vs. Polar stereographic for the resampled European region and the geos - geostationary projection of the full disc MSG dataset.
So we have to reproject a dataset.
Here we resample the satellite data to the elevation information by using `pyresample`_::

	>>> from pyresample import image
	>>> from pyresample import utils
	>>> dem_scene = msg_scene.project(tiffarea)
	Computing projection from meteosatseviri[-5567248.07417344 -5570248.47733926  5570248.47733926  5567248.07417344](3712, 3712) to laea_7238...
	Projecting channel VIS006 (0.635000μm)...
	Projecting channel VIS008 (0.810000μm)...
	Projecting channel IR_108 (10.800000μm)...
	Projecting channel IR_134 (13.400000μm)...
	Projecting channel IR_039 (3.920000μm)...
	Projecting channel WV_062 (6.250000μm)...
	Projecting channel WV_073 (7.350000μm)...
	Projecting channel IR_016 (1.640000μm)...
	Projecting channel IR_087 (8.700000μm)...
	Projecting channel IR_097 (9.660000μm)...
	Projecting channel IR_120 (12.000000μm)...
	Computing projection from meteosatseviri[-5568247.98732816 -5569248.12167703  5569248.12167703  5568247.98732816](11136, 11136) to laea_7238...
	Projecting channel HRV (0.700000μm)...

The satellite data for the middle European section looks like this (here the fog RGB composite has been displayed)::

	>>> fogdem = dem_scene.image.fog()
	>>> fogdem.show()

.. image:: ./fogpy_docu_example_4.png

We continue with cloud microphysical products for the selected satellite scene from a NetCDF 
file provided by the Climate Monitoring Satellite Application Facility (`CMSAF`_). 

.. _CMSAF: www.cmsaf.eu
.. _pyresample: https://github.com/pytroll/pyresample
.. _trollimage: http://trollimage.readthedocs.io/en/latest/

Therefore we extract the paramters and meta information from the NetCDF file that are required for geolocation and resampling::
 


	>>> cpp_file = '/media/nas/x21308/fog_db/result_{}_cpp.nc'.format(time.strftime("%Y%m%d%H%M"))
	>>> cpp = h5py.File(cpp_file, 'r')
	>>> proj4 = cpp.attrs["CMSAF_proj4_params"]
	>>> extent = cpp.attrs["CMSAF_area_extent"]
	>>> cot = list(cpp["cot"])[0] * 0.01
	>>> reff = list(cpp["reff"])[0] * 1.e-08
	>>> cwp = list(cpp["cwp"][:])[0] * 0.0002
	>>> area_id = 'CPP_cmsaf'
	>>> area_name = 'Gridded cloud physical properties from CMSAF'
	>>> proj_id = 'CPP_cmsaf'
	>>> x_size = cot.shape[0]
	>>> y_size = cot.shape[1]
	>>> cpp_area = utils.get_area_def(area_id, area_name, proj_id, proj4,
        		                      x_size, y_size, extent)

Afterwards the cloud optical depth (cod), effective droplet radius (reff) and the liquid water path (lwp) 
are extracted and resampled to the DEM projection again with the `pyresample`_ package::

	>>> cot_fd = image.ImageContainerQuick(cot, cpp_area)
	>>> reff_fd = image.ImageContainerQuick(reff, cpp_area)
	>>> cwp_fd = image.ImageContainerQuick(cwp, cpp_area)
	>>> cot_dem = cot_fd.resample(tiffarea)
	>>> reff_dem = reff_fd.resample(tiffarea)
	>>> cwp_dem = cwp_fd.resample(tiffarea)
	
Let's see how the data look like. We use the PyTroll package `trollimage`_ to 
visualize the cloud optical thickness product with automatic palettized colors in the range of 0 to 100::

	>>> from trollimage.colormap import set3
	>>> from trollimage.image import Image
	>>> img = Image(cot_dem.image_data, mode="L")
	>>> set3.set_range(0., 100.)
	>>> img.palettize(set3)
	>>> img.show()

.. image:: ./fogpy_docu_example_6.png

Creating a FLS product with fogpy
=================================

After we imported all required metadata we can continue with a fogpy composite.

.. note::
	Make sure that the fogpy composites are made available in the mpop.cfg! 

Add the following to the mpop.cfg file in the [composites] field. The config file can be found in the *PPP_CONFIG_DIR*::

	[composites]
	>>> module=fogpy.composites

Now all fogpy composites can be used directly in mpop. Let's try it with the *fls_day* composite.
This composite determine low clouds and ground fog cells from a satellite scene. 
It is limited to daytime because it requires channels in the visible spectrum to be successfully applicable. 
We create a fogpy composite for the resampled MSG scene.
Use the elevation and micro-physical parameters that we imported above as additionally input for the composite::

	>>> fls_img, fogmask = dem_scene.image.fls_day(elevation.image_data,
	>>>                                    	       cot_dem.image_data,
	>>>                                            reff_dem.image_data,
	>>>                                            cwp_dem.image_data)

You see that we don't have to import the fogpy package manually.
It's done automagically in the background after the mpop configuration.

The *fls_day* composite function returns two objects:
 
- An image of a selected channel (Default is the 10.8 IR channel) where only the detected ground fog cells are displayed
- An image for the fog mask

.. image:: ./fogpy_docu_example_10.png
 
The result image shows the area with potential ground fog calculated by the algorithm, fine.
But the remaining areas are missing... maybe a different visualization could be helpful.
We can improve the image output by colorize the fog mask and blending it over an overview composite using trollimage::

	>>> from trollimage.image import Image
	>>> from trollimage.colormap import Colormap
	>>> fogcol = Colormap((0., (0.0, 0.0, 0.8)),
   	>>> 	              (1., (250 / 255.0, 200 / 255.0, 40 / 255.0)))
	# Overlay fls image
	>>> fogmaskimg = Image(fogmask.channels[0], mode="L")
	>>> fogmaskimg.colorize(fogcol)
	>>> fogmaskimg.convert("RGBA")
	>>> alpha = np.zeros(fogmask.channels[0].shape)
	>>> alpha[fogmask.channels[0] == 1] = 0.5
	>>> fogmaskimg.putalpha(alpha)
	# Background overview composite
	>>> dem_overview = dem_scene.image.overview()
	>>> dem_fogimg = Image(dem_overview.channels, mode='RGB')
	>>> dem_fogimg.convert("RGBA")
	# Over blend fog mask
	>>> dem_fogimg.blend(fogmaskimg)
	>>> dem_fogimg.show()    	              
	>>> fls_img.show()

.. image:: ./fogpy_docu_example_11.png

As additional default, all successively applied filter outputs are saved as images with yellow colored fitler mask result values in the */tmp* directory.

Here are some example algorithm results for the given MSG scene. 
As describt above, the different masks are blendes over the overview RGB composite in yellow, except the right image where the fog RGB is in the background:

+----------------------------------------+----------------------------------------+----------------------------------------+
| .. image:: ./fogpy_docu_example_13.png | .. image:: ./fogpy_docu_example_12.png | .. image:: ./fogpy_docu_example_14.png |
+----------------------------------------+----------------------------------------+----------------------------------------+
|              Cloud mask                |               Low cloud mask           |         Low cloud mask + Fog RGB       |
+----------------------------------------+----------------------------------------+----------------------------------------+

It looks like the cloud mask works correctly and low cloud areas that are found by the algorithm fit quite good to the fog RGB yellowish areas.  