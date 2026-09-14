import os

# constants from miniz.h/miniz.c
data_descripter_size64 = 24
data_descripter_size32 = 16
mz_uint32_max = 0xFFFFFFFF

run_debug_asserts = os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1"
overall_storage = None
load_module_mapping: dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }