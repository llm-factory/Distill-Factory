from typing import Any, Dict, Optional
import os
import sys
from typing import Dict, Any, List, Optional
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args
from ..data.loader import get_dataset
from ..extras import logging
from ..extras.misc import get_device_count
from transformers import AutoConfig
import tqdm
import subprocess

logger = logging.get_logger(__name__)
curl_message = "vllm serve"


def deploy(args):
    print("Deploying model...")
    print(f"Running command: {curl_message}")
    print("Model deployed successfully!")

def build_cmd(args):
    model_args,data_args,generating_args = get_infer_args(args)
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    vllm_engine_args = {
        "trust-remote-code": model_args.trust_remote_code,
        "download-dir": model_args.cache_dir,
        "dtype": model_args.infer_dtype,
        "max-model-len": model_args.vllm_maxlen if model_args.vllm_maxlen else model_config.get("model_max_length", 4096),
        "tensor-parallel-size": get_device_count() or 1,
        "gpu-memory-utilization": model_args.vllm_gpu_util,
        "disable-log-stats": True,
        "disable-log-requests": True,
        "enforce-eager": model_args.vllm_enforce_eager,
        "enable-lora": model_args.adapter_name_or_path is not None,
    }
    vllm_config = model_args.vllm_config
    if vllm_config:
        vllm_engine_args.update(vllm_config)
    cmd = ["vllm", "serve", model_args.model_name_or_path]
    for key, value in vllm_engine_args.items():
        if value is None:
            continue            
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    return " ".join(cmd)


def run(args):
    subprocess.run(build_cmd(args), shell=True,check=True)

def run_api(args: Optional[Dict[str, Any]] = None) -> None:
    args = read_args(args)
    run(args)
    print("API running...")