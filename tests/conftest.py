"""pytest config"""

import os

import pytest


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
