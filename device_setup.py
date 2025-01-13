from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch

npu_available = False
cuda_available = False

# Globally import torch_npu if available
try:
    import torch_npu
except ImportError:
    torch_npu = None


def safe_get_available_device():
    global cuda_available, npu_available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda_available = True
        npu_available = False
    else:
        if torch_npu:
            device = torch.device("npu:0")
            torch_npu.npu.set_device(device.index)
            npu_available = torch_npu.npu.is_available()
            cuda_available = False
        else:
            device = torch.device("cpu")
            cuda_available = False
            npu_available = False
            raise RuntimeError("No available device found")
    return device


device = safe_get_available_device()


def safe_is_available():
    return device.type != "cpu"


def safe_device_count():
    if cuda_available:
        return torch.cuda.device_count()
    elif npu_available and torch_npu:
        return safe_device_count
    else:
        return 1


def safe_synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "npu" and torch_npu:
        safe_synchronize()
    else:
        raise RuntimeError("No available device found for synchronization")


@contextmanager
def safe_amp_autocast(enabled: Optional[bool] = None, **kwargs: Dict[str, Any]):
    """
    A context manager for automatic mixed precision (AMP) that works across different devices (CUDA/NPU).

    Args:
        enabled (Optional[bool]): If specified, determines whether autocast is enabled.
                                If None, autocast is enabled for GPU/NPU and disabled for CPU.
        **kwargs: Additional arguments to pass to the autocast context manager.
    """
    if enabled is None:
        enabled = device.type != "cpu"

    if device.type == "cuda":
        with torch.cuda.amp.autocast(enabled=enabled, **kwargs) as context:
            yield context
    elif device.type == "npu" and torch_npu:
        with safe_amp_autocast(enabled=enabled, **kwargs) as context:
            yield context
    else:
        # For CPU, we use a dummy context manager
        class DummyContext:
            def __init__(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        with DummyContext() as context:
            yield context


def safe_get_device_properties(device_index: int):
    """
    Get device properties for either CUDA or NPU device.

    Args:
        device_index (int): Index of the device to query

    Returns:
        DeviceProperties object containing device information
    """
    if cuda_available:
        return torch.cuda.get_device_properties(device_index)
    elif npu_available and torch_npu:
        return torch_npu.npu.get_device_properties(device_index)
    else:
        raise RuntimeError("No available device found for getting properties")


def safe_get_device_name(device_index: int) -> str:
    """
    Get device name for either CUDA or NPU device.

    Args:
        device_index (int): Index of the device to query

    Returns:
        str: Name of the device
    """
    props = safe_get_device_properties(device_index)
    return props.name
