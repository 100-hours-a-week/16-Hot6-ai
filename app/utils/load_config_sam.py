import os
from hydra import initialize_config_dir
from contextlib import contextmanager

@contextmanager
def hydra_config_context(config_path: str):
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with initialize_config_dir(config_dir=config_dir):
        yield config_name