# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union
from typing_extensions import Self
from transformers.training_args import _convert_str_dict

@dataclass
class VllmArguments:
    r"""
    Arguments pertaining to the vLLM worker.
    """

    vllm_maxlen: int = field(
        default=4096,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    vllm_config: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "Config to initialize the vllm engine. Please use JSON strings."},
    )


@dataclass
class ModelArguments(VllmArguments):
    r"""
    Arguments pertaining to which model we are going to infer using vllm.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models. Can also be the model name for api call"
        },
    ),
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    ),
    rope_scaling: Optional[Literal["linear", "dynamic", "yarn", "llama3"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    ),
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."},
    )
    base_url: Optional[str] = field(
        default=None,
        metadata={"help": "The base URL for api call. Avoid local api deploy if provided."},
    )
    api_key: Optional[str] = field(
        default=None,
        metadata={"help": "The API key for api call."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    model_max_length: Optional[int] = field(
        default=None,
        init=False,
        metadata={"help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."},
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={"help": "Whether use block diag attention or not, derived from `neat_packing`. Do not specify it."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    compute_dtype: Optional[torch.dtype] = field(
        default=None,
        init=False,
        metadata={"help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."},
    )
    device_map: Optional[Union[str, dict[str, Any]]] = field(
        default=None,
        init=False,
        metadata={"help": "Device map for model placement, derived from training stage. Do not specify it."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )   
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    
    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if isinstance(self.vllm_config, str) and self.vllm_config.startswith("{"):
            self.vllm_config = _convert_str_dict(json.loads(self.vllm_config))