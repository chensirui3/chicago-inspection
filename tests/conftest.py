"""Shared test fixtures. Tests marked ``data`` need the city's snapshot and skip without it."""

import pytest

from inspections import load


def pytest_collection_modifyitems(config, items):
    if load.data_available():
        return
    skip = pytest.mark.skip(reason="city data snapshot not found; run scripts/get_data.py")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)
