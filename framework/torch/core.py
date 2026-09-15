import io
import warnings
import tarfile
import sys
import os
import warnings
import importlib
import torch
import framework.torch.configure as configure
from typing import Any
from core import ObjRep, Convert, Assemble, convert
from framework.torch.reconstruct import *

from torch.types import FileLike
from torch.serialization import MAP_LOCATION, IS_WINDOWS, MAP_SHARED, MAGIC_NUMBER, PROTOCOL_VERSION
from torch.serialization import get_default_mmap_options

# Reuse the following funtions from PyTorch
_is_torchscript_zip = torch.serialization._is_torchscript_zip
_should_read_directly = torch.serialization._should_read_directly
_open_zipfile_reader = torch.serialization._open_zipfile_reader
_is_path = torch.serialization._is_path
_is_zipfile = torch.serialization._is_zipfile
_open_file_like = torch.serialization._open_file_like
_check_seekable = torch.serialization._check_seekable
# Is this a legacy checkpoint?
is_legacy = False

def generate(
        f: FileLike, 
        code_path: str,
        *, 
        indent: int = 4,
        pickle_file: str = "data.pkl", 
        **pickle_load_args: Any,
        ) -> Any:
    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"
    
    def persistent_load(saved_id):
        # Persistent load is an object
        return ObjRep(f"persistent_load({saved_id})")

    with _open_file_like(f, "rb") as opened_file:
        if _is_zipfile(opened_file):
            orig_position = opened_file.tell()
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                # Skip TorchScript files
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "received a zip file that looks like a TorchScript archive"
                        " (call 'torch.jit.load' directly to deserialize)",
                        UserWarning,
                        stacklevel=2,
                    )
                    opened_file.seek(orig_position)
                # Load the model and convert
                data_file = io.BytesIO(opened_zipfile.get_record(pickle_file))
                unpickler = Convert(data_file, indent=indent, **pickle_load_args)
                unpickler.persistent_load = persistent_load
                raw_code = unpickler.load()
                Assemble("torch", indent).output(raw_code, code_path)
                return
        
        _check_seekable(opened_file)
        f_should_read_directly = _should_read_directly(opened_file)
        
        if f_should_read_directly and opened_file.tell() == 0:
            # legacy_load requires that f has fileno()
            try:
                # We skip TAR format for now. May implement later if cases are found
                raise tarfile.TarError("Legacy TAR format (before PyTorch v1.6) is not safe and not supported!")
            except tarfile.TarError:
                if _is_zipfile(opened_file):
                    # .zip is used for torch.jit.save and will throw an un-pickling error here
                    raise RuntimeError(
                        f"{f} is a zip archive (did you mean to use torch.jit.load()?)"
                    ) from None
                # if not a tarfile, reset file offset and proceed
                opened_file.seek(0)

        raw_code = []

        # Magic number, protocol version, and system info
        raw_code.append(convert(opened_file, indent=indent, **pickle_load_args))
        raw_code.append(convert(opened_file, indent=indent, **pickle_load_args))
        raw_code.append(convert(opened_file, indent=indent, **pickle_load_args))

        # Main object
        unpickler = Convert(opened_file, indent=indent, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        raw_code.append(unpickler.load())

        # Deserialized storage keys
        raw_code.append(convert(opened_file, indent=indent, **pickle_load_args))

        # Offset. This is very important because it tells the program where the data starts
        offset = opened_file.tell() if f_should_read_directly else None
        raw_code.append(offset)

        Assemble("torch", indent).output(raw_code, code_path)

def execute(
    f: FileLike,
    code_path: str, 
    map_location: MAP_LOCATION = None,
    *,
    mmap: bool | None = None,
) -> Any:
    # Initialize the context
    ctx = configure.ExecutionContext()
    token = configure.set_context(ctx)
    try:
        # make flipping default BC-compatible
        if mmap is None:
            from torch.utils.serialization import config
            mmap = config.load.mmap

        # Dill is disabled
        # _check_dill_version(pickle_module)
        with _open_file_like(f, "rb") as opened_file:
            if _is_zipfile(opened_file):
                # TorchScript is no longer checked here because we do not generate code for it
                with _open_zipfile_reader(opened_file) as opened_zipfile:
                    if mmap:
                        if not _is_path(f):
                            raise ValueError(
                                "f must be a file path in order to use the mmap argument"
                            )
                        size = os.path.getsize(f)
                        if not IS_WINDOWS:
                            shared = get_default_mmap_options() == MAP_SHARED
                        else:
                            shared = False
                        ctx.overall_storage = torch.UntypedStorage.from_file(
                            os.fspath(f),
                            shared,
                            size,
                        )
                    configure.update(opened_zipfile, map_location)
                    ctx._serialization_tls.map_location = map_location
                    spec = importlib.util.spec_from_file_location("pypickle", code_path)
                    pypickle = importlib.util.module_from_spec(spec)
                    sys.modules["pypickle"] = pypickle
                    spec.loader.exec_module(pypickle)
                    result = pypickle.main()
                    ctx._serialization_tls.map_location = None
                    
                    torch._utils._validate_loaded_sparse_tensors()
                    torch._C._log_api_usage_metadata(
                        "torch.load.metadata", {"serialization_id": ctx.zip_file.serialization_id()}
                    )
                    return result
                
            if mmap:
                f_name = "" if not isinstance(f, str) else f"{f}, "
                raise RuntimeError(
                    "mmap can only be used with files saved with "
                    f"`torch.save({f_name}_use_new_zipfile_serialization=True), "
                    "please torch.save your checkpoint with this option in order to use mmap."
                )
            
            ctx.is_legacy = True
            _check_seekable(opened_file)
            f_should_read_directly = _should_read_directly(opened_file)
            if f_should_read_directly and opened_file.tell() == 0:
                # legacy_load requires that f has fileno()
                try:
                    tarfile.TarError("Legacy TAR format (before PyTorch v1.6) is not safe and not supported!")
                except tarfile.TarError:
                    if _is_zipfile(opened_file):
                        # .zip is used for torch.jit.save and will throw an un-pickling error here
                        raise RuntimeError(
                            f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                        ) from None
                    # if not a tarfile, reset file offset and proceed
                    opened_file.seek(0)

            spec = importlib.util.spec_from_file_location("pypickle", code_path)
            pypickle = importlib.util.module_from_spec(spec)
            sys.modules["pypickle"] = pypickle
            spec.loader.exec_module(pypickle)

            # Validation
            if pypickle.magic_number() != MAGIC_NUMBER:
                raise RuntimeError("Invalid magic number; corrupt file?")
            protocol_version = pypickle.protocol_version()
            if protocol_version != PROTOCOL_VERSION:
                raise RuntimeError(f"Invalid protocol version: {protocol_version}")

            # _sys_info = pypickle._sys_info()
            result = pypickle.main()
            deserialized_storage_keys = pypickle.deserialized_storage_keys()
            if torch._guards.active_fake_mode() is None and not ctx._serialization_tls.skip_data:
                offset = pypickle.OBJ_OFFSET
                for key in deserialized_storage_keys:
                    if key not in ctx.deserialized_objects:
                        raise AssertionError(
                            f"storage key {key!r} not found in deserialized_objects"
                        )
                    typed_storage = ctx.deserialized_objects[key]
                    typed_storage._untyped_storage._set_from_file(
                        opened_file,
                        offset,
                        f_should_read_directly,
                        torch._utils._element_size(typed_storage.dtype),
                    )
                    if offset is not None:
                        offset = opened_file.tell()

            torch._utils._validate_loaded_sparse_tensors()
            return result
    finally: 
        configure.clear_context(token)
