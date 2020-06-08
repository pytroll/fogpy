import pytest
import numpy
import datetime
import tempfile
import pathlib
from numpy import array
from xarray import Dataset, DataArray as xrda, open_dataset
from unittest import mock

import pkg_resources


@pytest.fixture
def fkattr():
    import pyresample
    return {
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
                3,
                3,
                (-155100.4363, -4441495.3795, 868899.5637, -3417495.3795)),
            'resolution': 3000.403165817,
            'calibration': 'reflectance',
            'polarization': None,
            'level': None,
            'modifiers': (),
            'ancillary_variables': []}


@pytest.fixture
def fogpy_inputs(fkattr):

    D = dict(
        ir108=array(
                [
                    [267.781, 265.75, 265.234],
                    [266.771, 265.75, 266.432],
                    [266.092, 266.771, 268.614],
                ]
            ),
        ir039=array(
                [
                    [274.07, 270.986, 269.281],
                    [273.335, 275.477, 277.236],
                    [277.023, 279.663, 279.663],
                ]
            ),
        vis008=array(
                [
                    [9.814, 10.652, 11.251],
                    [11.49, 13.884, 15.799],
                    [15.081, 16.158, 16.637],
                ]
            ),
        nir016=array(
                [
                    [9.439, 9.559, 10.156],
                    [11.47, 13.86, 16.011],
                    [15.055, 16.25, 16.967],
                ]
            ),
        vis006=array(
                [
                    [8.614, 9.215, 9.615],
                    [9.916, 11.519, 12.921],
                    [12.72, 13.422, 13.722],
                ]
            ),
        ir087=array(
                [
                    [265.139, 263.453, 262.67],
                    [264.225, 263.141, 263.608],
                    [263.453, 264.071, 266.338],
                ]
            ),
        ir120=array(
                [
                    [266.903, 265.208, 264.694],
                    [265.55, 264.522, 265.379],
                    [265.037, 265.037, 267.406],
                ]
            ),
        elev=array(
                [
                    [319.481, 221.918, 300.449],
                    [388.51, 501.519, 431.15],
                    [521.734, 520.214, 505.892],
                ]
            ),
        cot=array([[6.15, 10.98, 11.78], [13.92, 16.04, 7.93], [7.94, 10.01, 6.12]]),
        reff=array(
                [
                    [3.06e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 3.01e-06],
                    [3.01e-06, 3.01e-06, 9.32e-06],
                ]
            ),
        lwp=array([[0.013, 0.022, 0.024], [0.028, 0.032, 0.016], [0.016, 0.02, 0.038]]),
        lat=array(
                [
                    [50.669, 50.669, 50.67],
                    [50.614, 50.615, 50.616],
                    [50.559, 50.56, 50.561],
                ]
            ),
        lon=array([[6.437, 6.482, 6.528], [6.428, 6.474, 6.52], [6.42, 6.466, 6.511]]),
        cth=array(
                [
                    [4400.0, 4200.0, 4000.0],
                    [4200.0, 2800.0, 1200.0],
                    [1600.0, 1000.0, 800.0],
                ]
            ),
    )

    return {k: xrda(v, dims=("x", "y"), attrs={**fkattr, "name": k})
            for (k, v) in D.items()}


@pytest.fixture
def fogpy_inputs_seviri_cmsaf(fogpy_inputs):
    fogpy_inputs = fogpy_inputs.copy()
    trans = {"ir108": "IR_108",
             "ir039": "IR_039",
             "vis008": "VIS008",
             "nir016": "IR_016",
             "vis006": "VIS006",
             "ir087": "IR_087",
             "ir120": "IR_120",
             "lwp": "cwp"}
    for (k, v) in trans.items():
        fogpy_inputs[v] = fogpy_inputs.pop(k)
    return fogpy_inputs


@pytest.fixture
def fogpy_inputs_abi_nwcsaf(fogpy_inputs):
    fogpy_inputs = fogpy_inputs.copy()
    trans = {"ir108": "C14",
             "ir039": "C07",
             "vis008": "C03",
             "nir016": "C05",
             "vis006": "C02",
             "ir087": "C11",
             "ir120": "C15",
             "lwp": "cmic_lwp",
             "cot": "cmic_cot",
             "reff": "cmic_reff"}
    for (k, v) in trans.items():
        fogpy_inputs[v] = fogpy_inputs.pop(k)
    return fogpy_inputs


@pytest.fixture
def fog_comp_base():
    from fogpy.composites import FogCompositor
    return FogCompositor(name="fls_day")


@pytest.fixture
def fogpy_outputs():
    fls = numpy.ma.masked_array(
            numpy.arange(9, dtype="f4").reshape((3, 3)),
            (numpy.arange(9) % 2).astype("?").reshape((3, 3)))
    mask = (numpy.arange(9, dtype="f4") % 2).astype("?").reshape((3, 3))
    return (fls, mask)


@pytest.fixture
def comp_loader():
    """Get a compositor loader for loading fogpy composites."""
    from satpy.composites import CompositorLoader
    cpl = CompositorLoader(pkg_resources.resource_filename("fogpy", "etc/"))
    with mock.patch("requests.get") as rg:
        rg.return_value.content = b"12345"
        cpl.load_compositors(["seviri", "abi"])
    return cpl


@pytest.fixture
def fog_extra():
    return {
            "vcloudmask": numpy.ma.masked_array(
                data=[[True, True, True], [False, False, True], [True, False, False]],
                mask=numpy.zeros((3, 3), dtype="?"),
                fill_value=True),
            "cbh": numpy.zeros((3, 3)),
            "fbh": numpy.zeros((3, 3)),
            "lcth": numpy.full((3, 3), numpy.nan)}


@pytest.fixture
def fog_comp_day():
    from fogpy.composites import FogCompositorDay
    return FogCompositorDay(name="fls_day")


@pytest.fixture
def fog_comp_day_extra():
    from fogpy.composites import FogCompositorDayExtra
    return FogCompositorDayExtra(name="fls_day_extra")


@pytest.fixture
def fog_comp_night():
    from fogpy.composites import FogCompositorNight
    return FogCompositorNight(name="fls_night")


@pytest.fixture
def fog_intermediate_dataset(fog_extra, fogpy_outputs, fkattr):
    ds = Dataset(
            {k: xrda(v, dims=("y", "x"), attrs=fkattr) for (k, v) in
                fog_extra.items()},
            attrs=fkattr)
    (fls_day, fls_mask) = fogpy_outputs
    ds["fls_day"] = xrda(fls_day, dims=("y", "x"), attrs=fkattr)
    ds["fls_mask"] = xrda(fls_mask, dims=("y", "x"), attrs=fkattr)
    return ds


def test_convert_xr_to_ma(fogpy_inputs, fog_comp_base):
    fi_ma = fog_comp_base._convert_xr_to_ma(
            [fogpy_inputs["ir108"], fogpy_inputs["vis008"]])
    assert len(fi_ma) == 2
    assert all([isinstance(ma, numpy.ma.MaskedArray) for ma in fi_ma])
    assert numpy.array_equal(fi_ma[0].data, fogpy_inputs["ir108"].values)
    # try with some attributes missing
    ir108 = fogpy_inputs["ir108"].copy()
    del ir108.attrs["satellite_longitude"]
    del ir108.attrs["end_time"]
    fi_ma = fog_comp_base._convert_xr_to_ma([ir108])


def test_convert_ma_to_xr(fogpy_inputs, fog_comp_base, fogpy_outputs):
    conv = fog_comp_base._convert_ma_to_xr(
            [fogpy_inputs["ir108"], fogpy_inputs["vis008"]],
            *fogpy_outputs)
    assert len(conv) == len(fogpy_outputs)
    assert all([isinstance(c, xrda) for c in conv])
    numpy.testing.assert_array_equal(fogpy_outputs[0].data, conv[0].values)
    assert conv[0].attrs["sensor"] == fogpy_inputs["ir108"].attrs["sensor"]
    # check without mask
    conv = fog_comp_base._convert_ma_to_xr(
            [fogpy_inputs["ir108"], fogpy_inputs["vis008"]],
            *(fo.data for fo in fogpy_outputs))
    # try with some attributes missing
    ir108 = fogpy_inputs["ir108"].copy()
    del ir108.attrs["satellite_longitude"]
    del ir108.attrs["end_time"]
    fog_comp_base._convert_ma_to_xr([ir108])


def test_get_area_lat_lon(fogpy_inputs, fog_comp_base):
    (area, lat, lon) = fog_comp_base._get_area_lat_lon(
            [fogpy_inputs["ir108"], fogpy_inputs["vis008"]])
    assert area == fogpy_inputs["ir108"].area


def test_interim(fogpy_inputs_seviri_cmsaf, fogpy_inputs_abi_nwcsaf,
                 comp_loader, fogpy_outputs, fog_extra):
    fc_sev = comp_loader.get_compositor("_intermediate_fls_day", ["seviri"])
    fc_abi = comp_loader.get_compositor("_intermediate_fls_day", ["abi"])
    with mock.patch("fogpy.composites.Scene"), \
            mock.patch("fogpy.composites.DayFogLowStratusAlgorithm") as fcD:
        fcD.return_value.run.return_value = fogpy_outputs
        fcD.return_value.vcloudmask = fog_extra["vcloudmask"]
        fcD.return_value.cbh = fog_extra["cbh"]
        fcD.return_value.fbh = fog_extra["fbh"]
        fcD.return_value.lcth = fog_extra["lcth"]
        for (fc, fpi) in ((fc_sev, fogpy_inputs_seviri_cmsaf),
                          (fc_abi, fogpy_inputs_abi_nwcsaf)):
            ds = fc(
                    [fpi[k] for k in fc.attrs["prerequisites"]],
                    optional_datasets=[
                        fpi[k] for k in fc.attrs["optional_prerequisites"]
                        if k in fpi])
            assert isinstance(ds, Dataset)
            assert ds.data_vars.keys() == {
                    'vmask', 'fls_mask', 'fbh', 'cbh', 'fls_day', 'lcthimg'}
            numpy.testing.assert_equal(ds["cbh"].values, fcD.return_value.cbh)


def test_fog_comp_day(fog_comp_day, fog_intermediate_dataset):
    composite = fog_comp_day([fog_intermediate_dataset])
    assert isinstance(composite, xrda)


def test_fog_comp_day_extra(fog_comp_day_extra, fog_intermediate_dataset):
    composite = fog_comp_day_extra([fog_intermediate_dataset])
    assert isinstance(composite, Dataset)


def test_fog_comp_night(fog_comp_night, fogpy_inputs, fogpy_outputs):
    with mock.patch("fogpy.composites.NightFogLowStratusAlgorithm") as fcN:
        fcN.return_value.run.return_value = fogpy_outputs
        composite = fog_comp_night([fogpy_inputs["ir039"], fogpy_inputs["ir108"]])
    assert isinstance(composite, xrda)


def test_save_extras(fog_intermediate_dataset):
    from satpy import Scene
    from fogpy.composites import save_extras
    sc = Scene()
    sc["fls_day_extra"] = fog_intermediate_dataset
    with tempfile.TemporaryDirectory() as td:
        fn = pathlib.Path(td) / "tofu.nc"
        save_extras(sc, fn)
        with open_dataset(fn) as ds:
            ds.load()
            assert set(ds.data_vars.keys()) >= {
                    "vcloudmask", "fls_mask", "fbh", "cbh", "fls_day", "lcth"}
            numpy.testing.assert_array_equal(ds["fbh"].values, numpy.zeros((3, 3)))
