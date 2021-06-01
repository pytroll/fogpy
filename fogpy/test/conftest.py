# -*- coding: utf-8 -*-
"""conftest.py for fogpy."""

import os
import pytest
import pkg_resources


@pytest.fixture(scope="session", autouse=True)
def setUp(tmp_path_factory):
    for nm in {"XDG_CACHE_HOME", "XDG_DATA_HOME"}:
        os.environ[nm] = str(tmp_path_factory.mktemp(nm))
    os.environ["SATPY_CONFIG_PATH"] = pkg_resources.resource_filename(
            "fogpy", "etc/")
