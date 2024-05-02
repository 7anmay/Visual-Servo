from hydra import compose, initialize
import os
import sys

    
def load_config(config_name: str='config', overrides: list=[]):
    
    PYTHON_VERSION = sys.version_info[1]
    conf_dir = "conf"

    with initialize(config_path=conf_dir) if PYTHON_VERSION < 11 else initialize(version_base=None, config_path=conf_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg