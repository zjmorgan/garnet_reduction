"""pytest config"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # --all: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --all option to run")
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


@pytest.fixture(scope="session")
def has_datarepo():
    """Fixture that returns True if the datarepo_dir is available"""
    readme_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "garnet-data", "README.md")
    return os.path.exists(readme_data)


@pytest.fixture(scope="session")
def datarepo_dir():
    """Return the directory **absolute** paths for test data."""
    root_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "garnet-data")
    return root_data
