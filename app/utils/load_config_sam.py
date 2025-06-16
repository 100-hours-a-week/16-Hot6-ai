import os
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from contextlib import contextmanager

@contextmanager
def hydra_config_context(config_path: str):
    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    # ✅ 기존 Hydra 인스턴스 초기화
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir):
        yield config_name