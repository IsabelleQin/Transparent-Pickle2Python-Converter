import io
import warnings
import tarfile
import textwrap
import sys
import os
import warnings
import importlib
import torch
from typing import Any
from core.engine import *
from core.configs import *
from core.torch.support import *
from core.engine import find_class as find_class_pkl
from torch.serialization import StorageType
from torch.types import FileLike

is_legacy = False

def generate(
        f: FileLike, 
        code_path: str,
        *, 
        pickle_file="data.pkl", 
        **pickle_load_args: Any,
        ) -> Any:
    source_code = "from core.torch import _instantiate, build, find_class, get_extension, persistent_load\n"
    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"
    
    def persistent_load(saved_id):
        # Persistent load is an object
        return ObjRep(f"persistent_load({saved_id})")

    with open_file_like(f, "rb") as opened_file:
        if is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            with open_zipfile_reader(opened_file) as opened_zipfile:
                if is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "received a zip file that looks like a TorchScript archive"
                        " (call 'torch.jit.load' directly to deserialize)",
                        UserWarning,
                        stacklevel=2,
                    )
                    opened_file.seek(orig_position)
                # Load the data (which may in turn use `persistent_load` to load tensors)
                data_file = io.BytesIO(opened_zipfile.get_record(pickle_file))
                # source_code += "from core.torch import persistent_load_zip as persistent_load\n"
                unpickler = Convert(data_file, **pickle_load_args)
                # Needed for tensors where storage device and rebuild tensor device are
                # not connected (wrapper subclasses and tensors rebuilt using numpy)
                unpickler.persistent_load = persistent_load
                result = unpickler.load()
                source_code += "def main():\n"
                source_code += textwrap.indent(result.getvalue(), indent)
                with open(code_path, "w") as f:
                    f.write(source_code)
                return source_code
        
        check_seekable(opened_file)
        f_should_read_directly = should_read_directly(opened_file)
        
        if f_should_read_directly and opened_file.tell() == 0:
            # legacy_load requires that f has fileno()
            try:
                raise tarfile.TarError("Legacy TAR format (before PyTorch v1.6) is not safe and not supported!")
            except tarfile.TarError:
                if is_zipfile(opened_file):
                    # .zip is used for torch.jit.save and will throw an un-pickling error here
                    raise RuntimeError(
                        f"{f} is a zip archive (did you mean to use torch.jit.load()?)"
                    ) from None
                # if not a tarfile, reset file offset and proceed
                opened_file.seek(0)

        # source_code += "from core.torch import persistent_load_pkl as persistent_load\n"
        # Structure the result
        magic_number = convert(opened_file, **pickle_load_args)
        source_code += "def magic_number():\n"
        source_code += textwrap.indent(magic_number.getvalue(), indent)
        source_code += "\n"

        protocol_version = convert(opened_file, **pickle_load_args)
        source_code += "def protocol_version():\n"
        source_code += textwrap.indent(protocol_version.getvalue(), indent)
        source_code += "\n"

        _sys_info = convert(opened_file, **pickle_load_args)
        source_code += "def _sys_info():\n"
        source_code += textwrap.indent(_sys_info.getvalue(), indent)
        source_code += "\n"

        unpickler = Convert(opened_file, **pickle_load_args)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()
        source_code += "def main():\n"
        source_code += textwrap.indent(result.getvalue(), indent)
        source_code += "\n"

        deserialized_storage_keys = convert(opened_file, **pickle_load_args)
        source_code += "def deserialized_storage_keys():\n"
        source_code += textwrap.indent(deserialized_storage_keys.getvalue(), indent)
        source_code += "\n"

        offset = opened_file.tell() if f_should_read_directly else None
        # This is very important because it tells the program where the data starts
        source_code += f"OBJ_OFFSET = {offset}"
        with open(code_path, "w") as f:
            f.write(source_code)
        return source_code

def execute(
    f: FileLike,
    code_path: str, 
    map_location: MAP_LOCATION = None,
    *,
    mmap: bool | None = None,
    **pickle_load_args: Any,
) -> Any:
    # make flipping default BC-compatible
    if mmap is None:
        from torch.utils.serialization import config
        mmap = config.load.mmap

    # Dill is disabled
    # _check_dill_version(pickle_module)

    if "encoding" not in pickle_load_args:
        pickle_load_args["encoding"] = "utf-8"

    with open_file_like(f, "rb") as opened_file:
        if is_zipfile(opened_file):
            # TorchScript is no longer checked here because we do not generate code for it
            with open_zipfile_reader(opened_file) as opened_zipfile:
                global overall_storage
                if mmap:
                    if not is_path(f):
                        raise ValueError(
                            "f must be a file path in order to use the mmap argument"
                        )
                    size = os.path.getsize(f)
                    if not IS_WINDOWS:
                        shared = get_default_mmap_options() == MAP_SHARED
                    else:
                        shared = False
                    overall_storage = torch.UntypedStorage.from_file(
                        os.fspath(f),
                        shared,
                        size,
                    )
                zip_file = configure(opened_zipfile, map_location)
                global serialization_tls
                serialization_tls.map_location = map_location
                spec = importlib.util.spec_from_file_location("pypickle", code_path)
                pypickle = importlib.util.module_from_spec(spec)
                sys.modules["pypickle"] = pypickle
                spec.loader.exec_module(pypickle)
                result = pypickle.main()
                serialization_tls.map_location = None
                
                torch._utils._validate_loaded_sparse_tensors()
                torch._C._log_api_usage_metadata(
                    "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
                )
                return result
            
        if mmap:
            f_name = "" if not isinstance(f, str) else f"{f}, "
            raise RuntimeError(
                "mmap can only be used with files saved with "
                f"`torch.save({f_name}_use_new_zipfile_serialization=True), "
                "please torch.save your checkpoint with this option in order to use mmap."
            )
        
        global is_legacy
        is_legacy = True
        check_seekable(opened_file)
        f_should_read_directly = should_read_directly(opened_file)
        if f_should_read_directly and opened_file.tell() == 0:
            # legacy_load requires that f has fileno()
            try:
                tarfile.TarError("Legacy TAR format (before PyTorch v1.6) is not safe and not supported!")
            except tarfile.TarError:
                if is_zipfile(opened_file):
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
    
        _sys_info = pypickle._sys_info()
        result = pypickle.main()
        deserialized_storage_keys = pypickle.deserialized_storage_keys()
        if torch._guards.active_fake_mode() is None and not serialization_tls.skip_data:
            offset = pypickle.OBJ_OFFSET
            for key in deserialized_storage_keys:
                if key not in deserialized_objects:
                    raise AssertionError(
                        f"storage key {key!r} not found in deserialized_objects"
                    )
                typed_storage = deserialized_objects[key]
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

def find_class(mod_name, name, proto, fix_imports):
    if type(name) is str and "Storage" in name:
        try:
            return StorageType(name)
        except KeyError:
            pass
    if not is_legacy: # The following line is not implemented in legacy load
        mod_name = load_module_mapping.get(mod_name, mod_name)
    return find_class_pkl(mod_name, name, proto, fix_imports)


def persistent_load(saved_id):
    if is_legacy: return persistent_load_pkl(saved_id)
    else: return persistent_load_zip(saved_id)