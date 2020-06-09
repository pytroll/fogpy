import logging

import pytest
import unittest.mock


@unittest.mock.patch("requests.get")
def test_dl_dem(rg, tmp_path, caplog):
    from fogpy.utils import dl_dem
    rg.return_value.content = b"12345"

    with caplog.at_level(logging.INFO):
        dl_dem(tmp_path / "foo")
    assert f"Downloading https://zenodo.org/record/3885398/files/foo to {tmp_path / 'foo'!s}" in caplog.text
    assert (tmp_path / "foo").exists()
    assert (tmp_path / "foo").open(mode="rb").read() == b"12345"
    with pytest.raises(FileExistsError):
        dl_dem(tmp_path / "foo")
