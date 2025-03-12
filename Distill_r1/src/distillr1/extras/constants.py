import os
from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

AUDIO_PLACEHOLDER = os.environ.get("AUDIO_PLACEHOLDER", "<audio>")
SUPPORTED_MODELS = OrderedDict()
DATA_CONFIG = "dataset_info.json"
DEFAULT_TEMPLATE = defaultdict(str)
FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = os.environ.get("IMAGE_PLACEHOLDER", "<image>")
RUNNING_LOG = "running_log.txt"
LAYERNORM_NAMES = {"norm", "ln"}
MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}
MULTIMODAL_SUPPORTED_MODELS = set()

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}
VIDEO_PLACEHOLDER = os.environ.get("VIDEO_PLACEHOLDER", "<video>")

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"





class AttentionFunction(str, Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"


class EngineName(str, Enum):
    HF = "huggingface"
    VLLM = "vllm"


class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"

def register_model_group(
    models: dict[str, dict[DownloadSource, str]],
    template: Optional[str] = None,
    multimodal: bool = False,
) -> None:
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path
        if template is not None and (
            any(suffix in name for suffix in ("-Chat", "-Distill", "-Instruct")) or multimodal
        ):
            DEFAULT_TEMPLATE[name] = template
        if multimodal:
            MULTIMODAL_SUPPORTED_MODELS.add(name)

register_model_group(
    models={
        "Llama-3-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
        },
        "Llama-3-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B",
        },
        "Llama-3-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
        },
        "Llama-3-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B-Instruct",
        },
        "Llama-3-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
            DownloadSource.OPENMIND: "LlamaFactory/Llama3-Chinese-8B-Instruct",
        },
        "Llama-3-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-70B-Chinese-Chat",
        },
        "Llama-3.1-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B",
        },
        "Llama-3.1-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B",
        },
        "Llama-3.1-405B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B",
        },
        "Llama-3.1-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        },
        "Llama-3.1-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B-Instruct",
        },
        "Llama-3.1-405B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B-Instruct",
        },
        "Llama-3.1-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-8B-Chinese-Chat",
        },
        "Llama-3.1-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-70B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-70B-Chinese-Chat",
        },
        "Llama-3.2-1B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B",
        },
        "Llama-3.2-3B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B",
        },
        "Llama-3.2-1B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct",
        },
        "Llama-3.2-3B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B-Instruct",
        },
        "Llama-3.3-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.3-70B-Instruct",
        },
    },
    template="llama3",
)