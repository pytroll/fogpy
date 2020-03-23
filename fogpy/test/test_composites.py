import pytest
import numpy
import datetime
import functools
from numpy import array
from xarray import DataArray as xrda


@pytest.fixture
def fogpy_inputs():
    import pyresample
    fkattr = {
            'satellite_longitude': 0.0,
            'satellite_latitude': 0.0,
            'satellite_altitude': 35785831.0,
            'orbital_parameters': {
                'projection_longitude': 0.0,
                'projection_latitude': 0.0,
                'projection_altitude': 35785831.0,
                'satellite_nominal_longitude': 0.0,
                'satellite_nominal_latitude': 0.0,
                'satellite_actual_longitude': 0.0688189107392428,
                'satellite_actual_latitude': 0.1640766475684642,
                'satellite_actual_altitude': 35782769.05113167},
            'sensor': 'seviri',
            'platform_name': 'Meteosat-11',
            'georef_offset_corrected': True,
            'start_time': datetime.datetime(2020, 1, 6, 10, 0, 10, 604000),
            'end_time': datetime.datetime(2020, 1, 6, 10, 12, 43, 608000),
            'area': pyresample.AreaDefinition(
                "germ",
                "germ",
                None,
                {
                    'a': '6378144',
                    'b': '6356759',
                    'lat_0': '90',
                    'lat_ts': '50',
                    'lon_0': '5',
                    'no_defs': 'None',
                    'proj': 'stere',
                    'type': 'crs',
                    'units': 'm',
                    'x_0': '0',
                    'y_0': '0'},
                1024,
                1024,
                (-155100.4363, -4441495.3795, 868899.5637, -3417495.3795)),
            'name': 'VIS006',
            'resolution': 3000.403165817,
            'calibration': 'reflectance',
            'polarization': None,
            'level': None,
            'modifiers': (),
            'ancillary_variables': []}

    mk = functools.partial(xrda, dims=("x", "y"), attrs=fkattr)

    return dict(
        ir08=mk(
            array(
                [
                    [267.781, 265.75, 265.234],
                    [266.771, 265.75, 266.432],
                    [266.092, 266.771, 268.614],
                ]
            ),
        ),
        ir139=mk(
            array(
                [
                    [274.07, 270.986, 269.281],
                    [273.335, 275.477, 277.236],
                    [277.023, 279.663, 279.663],
                ]
            ),
        ),
        vis008=mk(
            array(
                [
                    [9.814, 10.652, 11.251],
                    [11.49, 13.884, 15.799],
                    [15.081, 16.158, 16.637],
                ]
            ),
        ),
        nir016=mk(
            array(
                [
                    [9.439, 9.559, 10.156],
                    [11.47, 13.86, 16.011],
                    [15.055, 16.25, 16.967],
                ]
            ),
        ),
        vis006=mk(
            array(
                [
                    [8.614, 9.215, 9.615],
                    [9.916, 11.519, 12.921],
                    [12.72, 13.422, 13.722],
                ]
            ),
        ),
        ir087=mk(
            array(
                [
                    [265.139, 263.453, 262.67],
                    [264.225, 263.141, 263.608],
                    [263.453, 264.071, 266.338],
                ]
            ),
        ),
        ir120=mk(
            array(
                [
                    [266.903, 265.208, 264.694],
                    [265.55, 264.522, 265.379],
                    [265.037, 265.037, 267.406],
                ]
            ),
        ),
        elev=mk(
            array(
                [
                    [319.481, 221.918, 300.449],
                    [388.51, 501.519, 431.15],
                    [521.734, 520.214, 505.892],
                ]
            ),
        ),
        cot=mk(
            array([[6.15, 10.98, 11.78], [13.92, 16.04, 7.93], [7.94, 10.01, 6.12]]),
        ),
        reff=mk(
            array(
                [
                    [3.06e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 9.32e-06],
                ]
            ),
        ),
        lwp=mk(
            array([[0.013, 0.022, 0.024], [0.028, 0.032, 0.016], [0.016, 0.02, 0.038]]),
        ),
        lat=mk(
            array(
                [
                    [50.669, 50.669, 50.67],
                    [50.614, 50.615, 50.616],
                    [50.559, 50.56, 50.561],
                ]
            ),
        ),
        lon=mk(
            array([[6.437, 6.482, 6.528], [6.428, 6.474, 6.52], [6.42, 6.466, 6.511]]),
        ),
        cth=mk(
            array(
                [
                    [4400.0, 4200.0, 4000.0],
                    [4200.0, 2800.0, 1200.0],
                    [1600.0, 1000.0, 800.0],
                ]
            ),
        ),
    )

@pytest.fixture
def fog_comp_base():
    from fogpy.composites import FogCompositor
    return FogCompositor(name="fls_day")

@pytest.fixture
def fogpy_outputs():
    fls = numpy.ma.masked_array([[1, 2], [3, 4]], [[True, False], [False, False]])
    mask = numpy.ma.masked_array([[True, False], [False, True]],
                           [[True, False], [False, False]])
    return (fls, mask)

def test_convert_projectables(fogpy_inputs, fog_comp_base):
    fi_ma = fog_comp_base._convert_projectables(
            [fogpy_inputs["ir08"], fogpy_inputs["vis008"]])
    assert len(fi_ma) == 2
    assert all([isinstance(ma, numpy.ma.MaskedArray) for ma in fi_ma])
    assert numpy.array_equal(fi_ma[0].data, fogpy_inputs["ir08"].values)

def test_convert_ma_to_xr(fogpy_inputs, fog_comp_base, fogpy_outputs):
    conv = fog_comp_base._convert_ma_to_xr(
            [fogpy_inputs["ir08"], fogpy_inputs["vis008"]],
            *fogpy_outputs)
    assert len(conv) == len(fogpy_outputs)
    assert all([isinstance(c, xrda) for c in conv])
    assert numpy.array_equal(fogpy_outputs[0].data, conv[0].values)
