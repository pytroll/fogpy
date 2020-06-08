import pytest
import unittest.mock


@unittest.mock.patch("requests.get")
def test_dl_dem(rg, tmp_path):
    from fogpy.utils import dl_dem
    rg.return_value.content = b"12345"

    dl_dem(tmp_path / "foo")
    assert (tmp_path / "foo").exists()
    assert (tmp_path / "foo").open(mode="rb").read() == b"12345"
    with pytest.raises(FileExistsError):
        dl_dem(tmp_path / "foo")
