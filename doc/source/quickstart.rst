============
 Quickstart
============

The package uses OOP extensively, to allow higher level metaobject handling.

For this tutorial, we will use a MSG scene for creating different 
fog products.

First example 
=============

We start with the PyTroll package mpop. This package provide all functionalities 
to import and calibrate a MSG scene from HRIT files. Therefore you should make sure 
that mpop is properly configured and all environment variables like *PPP_CONFIG_DIR* 
are set and the HRIT files are in the given search path. For more guidance look up 
in the `mpop`_ documentation

.. _mpop: http://mpop.readthedocs.io/en/latest/install.html#getting-the-files-and-installing-them/

Ok, let's get it on::

	>>> from datetime import datetime
	>>> from mpop.satellites import GeostationaryFactory
	
	>>> time = datetime(2017, 12, 06, 10, 0)

	>>>	msg_scene = GeostationaryFactory.create_scene(satname="meteosat",
	>>>                                                   satnumber='10',
	>>>                                                   instrument="seviri",
	>>>                                                   time_slot=time)
	>>> msg_scene.load()

We imported a MSG scene from  06. December 2017 and loaded all twelve channels into the scene object.

Now we want to look at the IR 10.8 channel::

	>>> msg_scene.image[10.8].show()

.. image:: ./fogpy_docu_example_1.png

Everything seems correctly imported. We see a full disk image. So lets see if we can resample it to a European region::

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

A lot of clouds are present over Europe. Let's test a fog RGB composite to find some low clouds:: 

	>>> fogimg = eu_scene.image.fog()
	>>> fogimg.show()

.. image:: ./fogpy_docu_example_3.png

The reddish and dark colored clouds represent cold and high altitude clouds, 
whereas the yellow-greenish color over northern Europe (England, Germany, France, Denmark) is a indication for low clouds and fog.

In the next step we want to create a fog and low stratus (FLS) composite for this imported scene