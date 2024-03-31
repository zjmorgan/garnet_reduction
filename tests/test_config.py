import os

from garnet.reduction import configuration

def test_load_config():

    config_path = os.path.join('data', 'CORELLI_normalization.config')

    assert os.path.exist(config_path)

    config = configuration.load_config(config_path)

    assert config.instrument == 'CORELLI'
    