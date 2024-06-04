"""Test for garnet-data repository access"""

import json
import os

import pytest


@pytest.mark.datarepo()
def test_data_repo(has_datarepo: bool, datarepo_dir: str):
    """Test for garnet-data"""
    if not has_datarepo:
        pytest.skip("Garnet-data repository is not available")

    # ensure we have access to the garnet-data repo
    assert os.path.exists(datarepo_dir)

    # ensure we can access the test file
    test_file = os.path.join(datarepo_dir, "test.txt")
    assert os.path.exists(test_file)

    # ensure we can read the contents of the file
    with open(test_file, "r") as f:
        test_data = json.load(f)

    assert test_data == {"a": 1, "b": 2, "c": 3}
