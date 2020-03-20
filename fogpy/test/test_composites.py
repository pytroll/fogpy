import pytest
import numpy
from numpy import array
from xarray import DataArray as xrda


@pytest.fixture
def fogpy_inputs():
    return dict(
        ir08=xrda(
            array(
                [
                    [267.781, 265.75, 265.234],
                    [266.771, 265.75, 266.432],
                    [266.092, 266.771, 268.614],
                ]
            ),
            dims=("x", "y"),
        ),
        ir139=xrda(
            array(
                [
                    [274.07, 270.986, 269.281],
                    [273.335, 275.477, 277.236],
                    [277.023, 279.663, 279.663],
                ]
            ),
            dims=("x", "y"),
        ),
        vis008=xrda(
            array(
                [
                    [9.814, 10.652, 11.251],
                    [11.49, 13.884, 15.799],
                    [15.081, 16.158, 16.637],
                ]
            ),
            dims=("x", "y"),
        ),
        nir016=xrda(
            array(
                [
                    [9.439, 9.559, 10.156],
                    [11.47, 13.86, 16.011],
                    [15.055, 16.25, 16.967],
                ]
            ),
            dims=("x", "y"),
        ),
        vis006=xrda(
            array(
                [
                    [8.614, 9.215, 9.615],
                    [9.916, 11.519, 12.921],
                    [12.72, 13.422, 13.722],
                ]
            ),
            dims=("x", "y"),
        ),
        ir087=xrda(
            array(
                [
                    [265.139, 263.453, 262.67],
                    [264.225, 263.141, 263.608],
                    [263.453, 264.071, 266.338],
                ]
            ),
            dims=("x", "y"),
        ),
        ir120=xrda(
            array(
                [
                    [266.903, 265.208, 264.694],
                    [265.55, 264.522, 265.379],
                    [265.037, 265.037, 267.406],
                ]
            ),
            dims=("x", "y"),
        ),
        elev=xrda(
            array(
                [
                    [319.481, 221.918, 300.449],
                    [388.51, 501.519, 431.15],
                    [521.734, 520.214, 505.892],
                ]
            ),
            dims=("x", "y"),
        ),
        cot=xrda(
            array([[6.15, 10.98, 11.78], [13.92, 16.04, 7.93], [7.94, 10.01, 6.12]]),
            dims=("x", "y"),
        ),
        reff=xrda(
            array(
                [
                    [3.06e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 9.32e-06],
                ]
            ),
            dims=("x", "y"),
        ),
        lwp=xrda(
            array([[0.013, 0.022, 0.024], [0.028, 0.032, 0.016], [0.016, 0.02, 0.038]]),
            dims=("x", "y"),
        ),
        lat=xrda(
            array(
                [
                    [50.669, 50.669, 50.67],
                    [50.614, 50.615, 50.616],
                    [50.559, 50.56, 50.561],
                ]
            ),
            dims=("x", "y"),
        ),
        lon=xrda(
            array([[6.437, 6.482, 6.528], [6.428, 6.474, 6.52], [6.42, 6.466, 6.511]]),
            dims=("x", "y"),
        ),
        cth=xrda(
            array(
                [
                    [4400.0, 4200.0, 4000.0],
                    [4200.0, 2800.0, 1200.0],
                    [1600.0, 1000.0, 800.0],
                ]
            ),
            dims=("x", "y"),
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
    assert numpy.array_equal(fogpy_outputs.data[0], conv[0].values)
