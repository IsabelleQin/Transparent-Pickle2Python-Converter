import os
import sys
import torch
import difflib
import warnings
from torch.types import Storage
from torch._sources import get_source_lines_and_file
from typing import cast
from torch.serialization import get_default_load_endianness, LoadEndianness, SourceChangeWarning

overall_storage = None
serialization_tls = torch.serialization._SerializationLocal()
_maybe_decode_ascii = torch.serialization._maybe_decode_ascii

def configure(opened_zipfile, map_location):
    global restore_location
    _get_restore_location = torch.serialization._get_restore_location
    restore_location = _get_restore_location(map_location)
    global zip_file
    zip_file = opened_zipfile

    # Aligned with _load, pre-configure 
    global loaded_storages, is_meta_map_location
    loaded_storages = {}
    _is_meta_location = torch.serialization._is_meta_location
    is_meta_map_location = _is_meta_location(map_location)
    
    global can_calculate_storage_offsets
    if zip_file.has_record(".format_version"):
        version = zip_file.get_record(".format_version")
        can_calculate_storage_offsets = version >= b"1"
    else: can_calculate_storage_offsets = False

    global byteorderdata
    # check if byteswapping is needed
    byteordername = "byteorder"
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b"little", b"big"]:
            raise ValueError("Unknown endianness type: " + byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError("Invalid load endianness type")

    global storage_alignment
    storage_alignment = 64
    if zip_file.has_record(".storage_alignment"):
        storage_alignment = int(zip_file.get_record(".storage_alignment"))

    if (
        not zip_file.has_record(byteordername)
        and get_default_load_endianness() is None
        and sys.byteorder == "big"
    ):
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn(
            "The default load endianness for checkpoints without a byteorder mark "
            "on big endian machines was changed from 'native' to 'little' endian, "
            "to avoid this behavior please use "
            "torch.serialization.set_default_load_endianness to set "
            "the desired default load endianness",
            UserWarning,
            stacklevel=2,
        )

    from torch.utils.serialization import config
    global calculate_storage_offsets
    calculate_storage_offsets = config.load.calculate_storage_offsets

    global current_offset, offsets
    current_offset = None
    offsets = dict()
    return zip_file


# Persistent load support for zip formats
def persistent_load_zip(saved_id):
    run_debug_asserts = os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1"
    # constants from miniz.h/miniz.c
    data_descripter_size64 = 24
    data_descripter_size32 = 16
    mz_uint32_max = 0xFFFFFFFF

    def _get_offset(key, name, numel):
        global current_offset, offsets
        if name in offsets:
            storage_offset = offsets[name]
            return storage_offset

        if current_offset is None:
            if key != "0":
                raise AssertionError(f"expected key '0', got {key!r}")
            current_offset = zip_file.get_record_offset(name)
            local_header_offset = zip_file.get_record_header_offset(name)
            storage_offset = current_offset
        else:
            storage_offset = zip_file.get_record_offset_no_read(
                current_offset, name, numel, storage_alignment
            )
            local_header_offset = current_offset

        # This is only actually needed for storages that have typed_storage._data_ptr() == 0
        # after being read. Otherwise persistent_load would never "re-call" load_tensor
        # for a given key.
        offsets[name] = storage_offset

        # Increment current_offset to offset where next zipfile header starts
        current_offset = storage_offset + numel
        # add size of data descriptor after payload
        if numel > 0:
            if local_header_offset >= mz_uint32_max or numel >= mz_uint32_max:
                current_offset += data_descripter_size64
            else:
                current_offset += data_descripter_size32

        return storage_offset

    def load_tensor(dtype, nbytes, key, location):
        name = f"data/{key}"
        if torch._guards.detect_fake_mode(None) is not None or is_meta_map_location:
            storage = torch.UntypedStorage(nbytes, device="meta")
            if can_calculate_storage_offsets:
                storage._checkpoint_offset = _get_offset(key, name, nbytes)
            else:
                storage._checkpoint_offset = zip_file.get_record_offset(name)
        elif serialization_tls.skip_data:
            storage = torch.UntypedStorage(nbytes)
        elif overall_storage is not None:
            if can_calculate_storage_offsets and calculate_storage_offsets:
                storage_offset = _get_offset(key, name, nbytes)
                if run_debug_asserts:
                    if storage_offset != zip_file.get_record_offset(name):
                        raise RuntimeError(
                            "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                            f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                            f"{zip_file.get_record_offset(name)}"
                        )
            else:
                storage_offset = zip_file.get_record_offset(name)
            storage = overall_storage[storage_offset : storage_offset + nbytes]
        else:
            if can_calculate_storage_offsets and run_debug_asserts:
                # This is debug code that we use to test the validity of
                # torch.utils.serialization.config.load.calculate_storage_offsets throughout CI
                storage_offset = _get_offset(key, name, nbytes)
                if storage_offset != zip_file.get_record_offset(name):
                    raise RuntimeError(
                        "This is a debug assert that was run as the `TORCH_SERIALIZATION_DEBUG` environment "
                        f"variable was set: Incorrect offset for {name}, got {storage_offset} expected "
                        f"{zip_file.get_record_offset(name)}"
                    )
            storage = (
                zip_file.get_storage_from_record(name, nbytes, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # swap here if byteswapping is needed
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        if is_meta_map_location:
            # Skip restore_location for meta map_location. Since we already created
            # a meta storage above, calling restore_location would just redundantly
            # call _meta_deserialize which creates another meta storage with the same
            # size.
            wrap_storage = storage
        elif torch._guards.detect_fake_mode(None) is None:
            wrap_storage = restore_location(storage, location)
        else:
            storage._fake_device = location
            wrap_storage = storage

        typed_storage = torch.storage.TypedStorage(
            wrap_storage=wrap_storage,
            dtype=dtype,
            _internal=True,
        )
        global loaded_storages
        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage
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

    global loaded_storages
    if key in loaded_storages:
        typed_storage = loaded_storages[key]
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

    global deserialized_objects
    deserialized_objects = {}
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

        if root_key not in deserialized_objects:
            if torch._guards.active_fake_mode() is not None:
                obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
            elif serialization_tls.skip_data:
                obj = cast(Storage, torch.UntypedStorage(nbytes))
                obj = restore_location(obj, location)
            else:
                obj = cast(Storage, torch.UntypedStorage(nbytes))
                obj._torch_load_uninitialized = True
                obj = restore_location(obj, location)
            # TODO: Once we decide to break serialization FC, we can
            # stop wrapping with TypedStorage
            typed_storage = torch.storage.TypedStorage(
                wrap_storage=obj, dtype=dtype, _internal=True
            )
            deserialized_objects[root_key] = typed_storage
        else:
            typed_storage = deserialized_objects[root_key]
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
            if view_key not in deserialized_objects:
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                deserialized_objects[view_key] = torch.storage.TypedStorage(
                    wrap_storage=typed_storage._untyped_storage[
                        offset_bytes : offset_bytes + view_size_bytes
                    ],
                    dtype=dtype,
                    _internal=True,
                )
            res = deserialized_objects[view_key]

        else:
            res = typed_storage
        return res
    else:
        raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    