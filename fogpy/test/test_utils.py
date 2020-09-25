# Copyright (c) 2017-2020 Fogpy developers

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
