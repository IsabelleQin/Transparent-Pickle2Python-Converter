# Maintain the reconstruction context
import sys
import warnings
import contextvars
import torch.serialization
from torch.serialization import get_default_load_endianness, LoadEndianness

_active_context = contextvars.ContextVar("active_context")

class ExecutionContext:
    def __init__(self):
        # Default values
        self.is_legacy = False
        self.overall_storage = None
        self.restore_location = None
        self.zip_file = None
        self.loaded_storages = {}
        self.deserialized_objects = {}
        self.is_meta_map_location = False
        self.can_calculate_storage_offsets = False
        self.byteorderdata = None
        self.storage_alignment = 64
        self.current_offset = None
        self.offsets = {}

        self._serialization_tls = torch.serialization._SerializationLocal()

        from torch.utils.serialization import config
        self.calculate_storage_offsets = config.load.calculate_storage_offsets

def get_context() -> ExecutionContext:
    """
        Helper function for persistent_load functions, return the current context.
    """
    try:
        return _active_context.get()
    except LookupError:
        raise RuntimeError("Attempted to load data without an active ConversionContext. "
                           "Ensure you are running within the converter runtime.")

def set_context(ctx: ExecutionContext):
    """
        Setup the context, called during execution.
    """
    return _active_context.set(ctx)

def clear_context(token):
    """Cleanup after the conversion is done."""
    _active_context.reset(token)

def update(opened_zipfile, map_location):
    ctx = get_context()
    _get_restore_location = torch.serialization._get_restore_location
    ctx.restore_location = _get_restore_location(map_location)
    ctx.zip_file = opened_zipfile

    # Aligned with _load, pre-configure 
    _is_meta_location = torch.serialization._is_meta_location
    ctx.is_meta_map_location = _is_meta_location(map_location)
    
    if ctx.zip_file.has_record(".format_version"):
        version = ctx.zip_file.get_record(".format_version")
        ctx.can_calculate_storage_offsets = version >= b"1"
    else: ctx.can_calculate_storage_offsets = False

    # check if byteswapping is needed
    byteordername = "byteorder"
    if ctx.zip_file.has_record(byteordername):
        ctx.byteorderdata = ctx.zip_file.get_record(byteordername)
        if ctx.byteorderdata not in [b"little", b"big"]:
            raise ValueError("Unknown endianness type: " +ctx.byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        ctx.byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        ctx.byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError("Invalid load endianness type")

    if ctx.zip_file.has_record(".storage_alignment"):
        ctx.storage_alignment = int(ctx.zip_file.get_record(".storage_alignment"))

    if (
        not ctx.zip_file.has_record(byteordername)
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