import gc
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Sequence, Tuple, Union
import transformers
import torch
from transformers import InfNanRemoveLogitsProcessor, LogitsProcessorList
from transformers.dynamic_module_utils import get_relative_imports
from transformers.utils.versions import require_version
from transformers.utils import (
    is_torch_bf16_gpu_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from .packages import is_transformers_version_greater_than
from . import logging

logger = logging.get_logger(__name__)

_is_fp16_available = is_torch_npu_available() or is_torch_cuda_available()
try:
    _is_bf16_available = is_torch_bf16_gpu_available() or (is_torch_npu_available() and torch.npu.is_bf16_supported())
except Exception:
    _is_bf16_available = False
    
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


def get_logits_processor() -> "LogitsProcessorList":
    r"""Get logits processor that removes NaN and Inf logits."""
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor

def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def skip_check_imports() -> None:
    r"""Avoid flash attention import error in custom model files."""
    if not is_env_enabled("FORCE_CHECK_IMPORTS"):
        transformers.dynamic_module_utils.check_imports = get_relative_imports
        

def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    if (not use_modelscope() and not use_openmind()) or os.path.exists(model_args.model_name_or_path):
        return model_args.model_name_or_path

    if use_modelscope():
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import snapshot_download  # type: ignore

        revision = "master" if model_args.model_revision == "main" else model_args.model_revision
        return snapshot_download(
            model_args.model_name_or_path,
            revision=revision,
            cache_dir=model_args.cache_dir,
        )

    if use_openmind():
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind.utils.hub import snapshot_download  # type: ignore

        return snapshot_download(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
        )
        
        
def get_current_device() -> "torch.device":
    r"""Get the current available device."""
    if is_torch_xpu_available():
        device = "xpu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_npu_available():
        device = "npu:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_mps_available():
        device = "mps:{}".format(os.environ.get("LOCAL_RANK", "0"))
    elif is_torch_cuda_available():
        device = "cuda:{}".format(os.environ.get("LOCAL_RANK", "0"))
    else:
        device = "cpu"

    return torch.device(device)


def infer_optim_dtype(model_dtype: "torch.dtype") -> "torch.dtype":
    r"""Infer the optimal dtype according to the model_dtype and device compatibility."""
    if _is_bf16_available and model_dtype == torch.bfloat16:
        return torch.bfloat16
    elif _is_fp16_available:
        return torch.float16
    else:
        return torch.float32

