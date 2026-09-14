import os
import torch
from os import path
from core.nemo.support import Restore
from nemo.core.classes import ModelPT
from nemo.utils.app_state import AppState
from typing import Optional, Union
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.trainer.trainer import Trainer
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils.msc_utils import is_multistorageclient_url

def generate(restore_path: str, code_path: str):
    return Restore().generate(restore_path, code_path)

def execute(
    cls,
    restore_path: str,
    code_path: str,
    override_config_path: Optional[Union[OmegaConf, str]] = None,
    map_location: Optional[torch.device] = None,
    strict: bool = True,
    return_config: bool = False,
    save_restore_connector: SaveRestoreConnector = None,
    trainer: Optional[Trainer] = None,
    validate_access_integrity: bool = True,
):
    """
    Restores model instance (weights and configuration) from .nemo file.

    Returns:
        An instance of type cls or its underlying config (if return_config is set).
    """
    # Fix it
    save_restore_connector = Restore()
    if is_multistorageclient_url(restore_path):
        raise NotImplementedError("import_multistorageclient is not supported for now!")
    else:
        if save_restore_connector.model_extracted_dir is None:
            restore_path = os.path.abspath(os.path.expanduser(restore_path))
        else:
            restore_path = os.path.abspath(os.path.expanduser(save_restore_connector.model_extracted_dir))

        if not path.exists(restore_path):
            raise FileNotFoundError(f"Can't find {restore_path}")

    app_state = AppState()
    app_state.model_restore_path = restore_path

    cls.update_save_restore_connector(save_restore_connector)
    instance = cls._save_restore_connector.restore_from(
        cls,
        restore_path,
        code_path,
        override_config_path,
        map_location,
        strict,
        return_config,
        trainer,
        validate_access_integrity,
    )
    if isinstance(instance, ModelPT):
        instance._save_restore_connector = save_restore_connector

    return instance