import gc
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Sequence, Tuple, Union

import torch
from transformers.utils.versions import require_version
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from .packages import is_transformers_version_greater_than
from . import logging

logger = logging.get_logger(__name__)
def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()
        
        
def check_version(requirement: str, mandatory: bool = False) -> None:
    r"""
    Optionally checks the package version.
    """
    if is_env_enabled("DISABLE_VERSION_CHECK") and not mandatory:
        logger.warning_rank0_once("Version checking has been disabled, may lead to unexpected behaviors.")
        return

    if mandatory:
        hint = f"To fix: run `pip install {requirement}`."
    else:
        hint = f"To fix: run `pip install {requirement}` or set `DISABLE_VERSION_CHECK=1` to skip this check."

    require_version(requirement, hint)
        
def check_dependencies() -> None:
    r"""
    Checks the version of the required packages.
    """
    check_version("transformers>=4.41.2,<=4.49.0,!=4.46.0,!=4.46.1,!=4.46.2,!=4.46.3,!=4.47.0,!=4.47.1,!=4.48.0")
    check_version("datasets>=2.16.0,<=3.2.0")
    check_version("accelerate>=0.34.0,<=1.2.1")
    check_version("peft>=0.11.1,<=0.12.0")
    check_version("trl>=0.8.6,<=0.9.6")
    if is_transformers_version_greater_than("4.46.0") and not is_transformers_version_greater_than("4.48.1"):
        logger.warning_rank0_once("There are known bugs in transformers v4.46.0-v4.48.0, please use other versions.")
def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""
    Checks if the environment variable is enabled.
    """
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]

def use_modelscope() -> bool:
    return is_env_enabled("USE_MODELSCOPE_HUB")


def use_openmind() -> bool:
    return is_env_enabled("USE_OPENMIND_HUB")


def get_device_count() -> int:
    r"""
    Gets the number of available GPU or NPU devices.
    """
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0