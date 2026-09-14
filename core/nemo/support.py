import os
from os import path
import torch
import logging
import tempfile
from nemo.utils.app_state import AppState
from pathlib import Path
from core.torch import generate, execute
from contextlib import contextmanager, nullcontext
from typing import Optional, Union
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.trainer.trainer import Trainer
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

class Restore(SaveRestoreConnector):
    def _determine_dir(self):
        # Determine if we should use a pre-extracted directory
        use_extracted_dir = self.model_extracted_dir is not None and os.path.isdir(self.model_extracted_dir)

        if use_extracted_dir:
            logging.info(f"Restoration will occur within pre-extracted directory : " f"`{self.model_extracted_dir}`.")

        # Use nullcontext if we have an extracted dir, otherwise create a temp directory
        dir_context = nullcontext(self.model_extracted_dir) if use_extracted_dir else tempfile.TemporaryDirectory()

        return use_extracted_dir, dir_context

    
    def generate(self, restore_path: str, code_path: str):
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()
        # Register the output path and get the absolute path
        code_path = Path(cwd).joinpath(code_path)
        app_state = AppState()
        use_extracted_dir, dir_context = self._determine_dir()

        with dir_context as tmpdir:
            try:
                if not use_extracted_dir:
                    # Extract the nemo file into the temporary directory
                    filter_fn = None
                    members = self._filtered_tar_info(restore_path, filter_fn=filter_fn)
                    self._unpack_nemo_file(path2file=restore_path, out_folder=tmpdir, members=members)

                # Change current working directory to
                os.chdir(tmpdir)
                # Disable config only here (Enable during execution?)
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                else:
                    model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                # add load_state_dict override
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                
                try:
                    source_code = generate(model_weights, code_path)
                except Exception as e:
                    logging.error(f"Failed to convert checkpoint: {e}")
                    raise e
            finally:
                os.chdir(cwd)
        return source_code


    def load_config_and_state_dict(
        self,
        calling_cls,
        restore_path: str,
        code_path: str,
        override_config_path: Optional[Union[OmegaConf, str]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
        return_config: bool = False,
        trainer: Trainer = None,
        validate_access_integrity: bool = True,
    ):
        # Get path where the command is executed - the artifacts will be "retrieved" there
        # (original .nemo behavior)
        cwd = os.getcwd()

        if map_location is None:
            if torch.cuda.is_available():
                map_location = torch.device('cuda')
            else:
                map_location = torch.device('cpu')

        app_state = AppState()
        use_extracted_dir, dir_context = self._determine_dir()
        with dir_context as tmpdir:
            try:
                if not use_extracted_dir:
                    # Extract the nemo file into the temporary directory
                    filter_fn = None
                    if return_config:
                        filter_fn = lambda name: '.yaml' in name
                    members = self._filtered_tar_info(restore_path, filter_fn=filter_fn)
                    self._unpack_nemo_file(path2file=restore_path, out_folder=tmpdir, members=members)

                # Change current working directory to
                os.chdir(tmpdir)
                if override_config_path is None:
                    config_yaml = self.model_config_yaml
                else:
                    # can be str path or OmegaConf / DictConfig object
                    config_yaml = override_config_path
                if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                    conf = OmegaConf.load(config_yaml)
                else:
                    conf = config_yaml
                    if override_config_path is not None:
                        # Resolve the override config
                        conf = OmegaConf.to_container(conf, resolve=True)
                        conf = OmegaConf.create(conf)
                # If override is top level config, extract just `model` from it
                if 'model' in conf:
                    conf = conf.model

                if return_config:
                    instance = conf
                    return instance
                else:
                    if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                        model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                    else:
                        model_weights = os.path.join(tmpdir, self.model_weights_ckpt)
                OmegaConf.set_struct(conf, True)
                os.chdir(cwd)
                # get the class
                calling_cls._set_model_restore_state(is_being_restored=True, folder=tmpdir)
                instance = calling_cls.from_config_dict(config=conf, trainer=trainer)
                instance = instance.to(map_location)
                # add load_state_dict override
                if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
                    model_weights = self._inject_model_parallel_rank_for_ckpt(tmpdir, self.model_weights_ckpt)
                state_dict = execute(model_weights, code_path, map_location=map_location)
            finally:
                os.chdir(cwd)

        return (conf, instance, state_dict)

    def restore_from(
            self,
            calling_cls,
            restore_path: str,
            code_path: str,
            override_config_path: Optional[Union[OmegaConf, str]] = None,
            map_location: Optional[torch.device] = None,
            strict: bool = True,
            return_config: bool = False,
            trainer: Trainer = None,
            validate_access_integrity: bool = True,
        ):
            # Get path where the command is executed - the artifacts will be "retrieved" there
            # (original .nemo behavior)
            loaded_params = self.load_config_and_state_dict(
                calling_cls,
                restore_path,
                code_path,
                override_config_path,
                map_location,
                strict,
                return_config,
                trainer,
                validate_access_integrity,
            )
            if not isinstance(loaded_params, tuple) or return_config is True:
                return loaded_params
            conf, instance, state_dict = loaded_params
            state_dict = self.modify_state_dict(conf, state_dict)
            self.load_instance_with_state_dict(instance, state_dict, strict)
            logging.info(f'Model {instance.__class__.__name__} was successfully restored from {restore_path}.')
            return instance