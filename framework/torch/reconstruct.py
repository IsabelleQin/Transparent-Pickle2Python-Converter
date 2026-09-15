import os
import sys
import torch
import difflib
import warnings
from torch.types import Storage
from torch._sources import get_source_lines_and_file
from typing import cast
import framework.torch.configure as configure
from torch.serialization import SourceChangeWarning
from torch.serialization import StorageType
from core import find_class as find_class_pkl

_maybe_decode_ascii = torch.serialization._maybe_decode_ascii

# Persistent load support for zip formats
def persistent_load_zip(saved_id):
    ctx = configure.get_context()

    def _get_offset(key, name, numel):
        # constants from miniz.h/miniz.c
        data_descripter_size64 = 24
        data_descripter_size32 = 16
        mz_uint32_max = 0xFFFFFFFF
        # ctx = configure.get_context()
        if name in ctx.offsets:
            storage_offset = ctx.offsets[name]
            return storage_offset

        if ctx.current_offset is None:
            if key != "0":
                raise AssertionError(f"expected key '0', got {key!r}")
            ctx.current_offset = ctx.zip_file.get_record_offset(name)
            local_header_offset = ctx.zip_file.get_record_header_offset(name)
            storage_offset = ctx.current_offset
        else:
            storage_offset = ctx.zip_file.get_record_offset_no_read(
                ctx.current_offset, name, numel, ctx.storage_alignment
            )
            local_header_offset = ctx.current_offset

        # This is only actually needed for storages that have typed_storage._data_ptr() == 0
        # after being read. Otherwise persistent_load would never "re-call" load_tensor
        # for a given key.
        ctx.offsets[name] = storage_offset

        # Increment current_offset to offset where next zipfile header starts
        ctx.current_offset = storage_offset + numel
        # add size of data descriptor after payload
        if numel > 0:
            if local_header_offset >= mz_uint32_max or numel >= mz_uint32_max:
                ctx.current_offset += data_descripter_size64
            else:
                ctx.current_offset += data_descripter_size32

        return storage_offset

    def load_tensor(dtype, nbytes, key, location):
        run_debug_asserts = os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1"
        # ctx = configure.get_context()
        name = f"data/{key}"
        if torch._guards.detect_fake_mode(None) is not None or ctx.is_meta_map_location:
            storage = torch.UntypedStorage(nbytes, device="meta")
            if ctx.can_calculate_storage_offsets:
                storage._checkpoint_offset = _get_offset(key, name, nbytes)
            else:
                storage._checkpoint_offset = ctx.zip_file.get_record_offset(name)
        elif ctx._serialization_tls.skip_data:
            storage = torch.UntypedStorage(nbytes)
        elif ctx.overall_storage is not None:
            if ctx.can_calculate_storage_offsets and ctx.calculate_storage_offsets:
                storage_offset = _get_offset(key, name, nbytes)
                if run_debug_asserts:
                    if storage_offset != ctx.zip_file.get_record_offset(name):
                        raise RuntimeError(
                            "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                            f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                            f"{ctx.zip_file.get_record_offset(name)}"
                        )
            else:
                storage_offset = ctx.zip_file.get_record_offset(name)
            storage = ctx.overall_storage[storage_offset : storage_offset + nbytes]
        else:
            if ctx.can_calculate_storage_offsets and run_debug_asserts:
                # This is debug code that we use to test the validity of
                # torch.utils.serialization.config.load.calculate_storage_offsets throughout CI
                storage_offset = _get_offset(key, name, nbytes)
                if storage_offset != ctx.zip_file.get_record_offset(name):
                    raise RuntimeError(
                        "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                        f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                        f"{ctx.zip_file.get_record_offset(name)}"
                    )
            storage = (
                ctx.zip_file.get_storage_from_record(name, nbytes, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # swap here if byteswapping is needed
        if ctx.byteorderdata is not None:
            if ctx.byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        if ctx.is_meta_map_location:
            # Skip restore_location for meta map_location. Since we already created
            # a meta storage above, calling restore_location would just redundantly
            # call _meta_deserialize which creates another meta storage with the same
            # size.
            wrap_storage = storage
        elif torch._guards.detect_fake_mode(None) is None:
            wrap_storage = ctx.restore_location(storage, location)
        else:
            storage._fake_device = location
            wrap_storage = storage

        typed_storage = torch.storage.TypedStorage(
            wrap_storage=wrap_storage,
            dtype=dtype,
            _internal=True,
        )
        if typed_storage._data_ptr() != 0:
            ctx.loaded_storages[key] = typed_storage
        return typed_storage

    if not isinstance(saved_id, tuple):
        raise AssertionError(
            f"saved_id must be a tuple, got {type(saved_id).__name__}"
        )
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]

    if typename != "storage":
        raise AssertionError(
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        )
    
    storage_type, key, location, numel = data
    if storage_type is torch.UntypedStorage:
        dtype = torch.uint8
    else:
        dtype = storage_type.dtype

    if key in ctx.loaded_storages:
        typed_storage = ctx.loaded_storages[key]
    else:
        nbytes = numel * torch._utils._element_size(dtype)
        typed_storage = load_tensor(
            dtype, nbytes, key, _maybe_decode_ascii(location)
        )
    return typed_storage

# Persistent load support for legacy pickle formats
def persistent_load_pkl(saved_id):
    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading.",
                stacklevel=2,
            )
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + ".patch"
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except OSError:
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            warnings.warn(msg, SourceChangeWarning, stacklevel=2)
    ctx = configure.get_context()
    ctx.deserialized_objects = {}
    if not isinstance(saved_id, tuple):
        raise AssertionError(
            f"saved_id must be a tuple, got {type(saved_id).__name__}"
        )
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]

    if typename == "module":
        # Ignore containers that don't have any sources saved
        if all(data[1:]):
            _check_container_source(*data)
        return data[0]
    elif typename == "storage":
        storage_type, root_key, location, numel, view_metadata = data
        location = _maybe_decode_ascii(location)
        dtype = storage_type.dtype
        nbytes = numel * torch._utils._element_size(dtype)

        if root_key not in ctx.deserialized_objects:
            if torch._guards.active_fake_mode() is not None:
                obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
            elif ctx._serialization_tls.skip_data:
                obj = cast(Storage, torch.UntypedStorage(nbytes))
                obj = ctx.restore_location(obj, location)
            else:
                obj = cast(Storage, torch.UntypedStorage(nbytes))
                obj._torch_load_uninitialized = True
                obj = ctx.restore_location(obj, location)
            # TODO: Once we decide to break serialization FC, we can
            # stop wrapping with TypedStorage
            typed_storage = torch.storage.TypedStorage(
                wrap_storage=obj, dtype=dtype, _internal=True
            )
            ctx.deserialized_objects[root_key] = typed_storage
        else:
            typed_storage = ctx.deserialized_objects[root_key]
            if typed_storage._data_ptr() == 0:
                typed_storage = torch.storage.TypedStorage(
                    device=typed_storage._untyped_storage.device,
                    dtype=dtype,
                    _internal=True,
                )

        if view_metadata is not None:
            view_key, offset, view_size = view_metadata
            offset_bytes = offset * torch._utils._element_size(dtype)
            view_size_bytes = view_size * torch._utils._element_size(dtype)
            if view_key not in ctx.deserialized_objects:
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                ctx.deserialized_objects[view_key] = torch.storage.TypedStorage(
                    wrap_storage=typed_storage._untyped_storage[
                        offset_bytes : offset_bytes + view_size_bytes
                    ],
                    dtype=dtype,
                    _internal=True,
                )
            res = ctx.deserialized_objects[view_key]
        else:
            res = typed_storage
        return res
    else:
        raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

load_module_mapping: dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }
def find_class(mod_name, name, proto, fix_imports):
    ctx = configure.get_context()
    if type(name) is str and "Storage" in name:
        try:
            return StorageType(name)
        except KeyError:
            pass
    if not ctx.is_legacy: # The following line is not implemented in legacy load
        mod_name = load_module_mapping.get(mod_name, mod_name)
    return find_class_pkl(mod_name, name, proto, fix_imports)


def persistent_load(saved_id):
    ctx = configure.get_context()
    if ctx.is_legacy: return persistent_load_pkl(saved_id)
    else: return persistent_load_zip(saved_id)