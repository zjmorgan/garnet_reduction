"""pytest config"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "resources_intensive" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def has_sns_mount():
    """Fixture that returns True if the SNS data mount (e.g CORELLI shared) is available"""
    sns_dir = "/SNS/CORELLI/shared/"
    return os.path.exists(sns_dir)


@pytest.fixture(scope="session")
def has_hfir_mount():
    """Fixture that returns True if the HFIR data mount (e.g HB3A shared) is available"""
    hfir_dir = "/HFIR/HB3A/shared/"
    return os.path.exists(hfir_dir)
