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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from transformers import HfArgumentParser
from ..extras import logging
from ..extras.misc import check_dependencies, check_version, is_env_enabled
from .data_args import DataArguments
from .generating_args import GeneratingArguments
from .model_args import ModelArguments
from .distill_args import DistillArguments
from .fintuning_args import FinetuningArguments
from .curation_args import CurationArguments

logger = logging.get_logger(__name__)

check_dependencies()

_INFER_ARGS = [ModelArguments, DataArguments, GeneratingArguments,DistillArguments,FinetuningArguments]
_INFER_CLS = Tuple[List[ModelArguments], DataArguments, GeneratingArguments,DistillArguments,FinetuningArguments]
_ORIGIN_INFER_CLS = Tuple[ModelArguments, DataArguments, GeneratingArguments,DistillArguments,FinetuningArguments]

_CURATION_ARGS = [CurationArguments]
_CURATION_CLS = Tuple[CurationArguments]

def read_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> Union[Dict[str, Any], List[str]]:
    r"""
    Gets arguments from the command line or a config file.
    """
    if args is not None:
        return args

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return yaml.safe_load(Path(sys.argv[1]).absolute().read_text())
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return json.loads(Path(sys.argv[1]).absolute().read_text())
    else:
        return sys.argv[1:]


def _parse_args(
    parser: "HfArgumentParser", args: Optional[Union[Dict[str, Any], List[str]]] = None, allow_extra_keys: bool = False
) -> Tuple[Any]:
    args = read_args(args)
    chatmodel_args = args.pop("chat", None)
    if isinstance(args, dict):
        parsed_args = parser.parse_dict(args, allow_extra_keys=allow_extra_keys)
        if chatmodel_args is not None:
            chatmodel_args_list = []
            for chatmodel_arg in chatmodel_args:
                model_parser = HfArgumentParser([ModelArguments,GeneratingArguments,DataArguments])
                client_arg = model_parser.parse_dict(chatmodel_arg)
                model_args, generating_args,data_args = client_arg
                chatmodel_args_list.append((model_args, generating_args,data_args))
            return (chatmodel_args_list,) + parsed_args[1:] # leave out the default model_args
        else:
            return ([parsed_args[0]],) + parsed_args[1:]

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(args=args, return_remaining_strings=True)

    if unknown_args and not allow_extra_keys:
        print(parser.format_help())
        print(f"Got unknown args, potentially deprecated arguments: {unknown_args}")
        raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {unknown_args}")
    return tuple(parsed_args)


def _check_extra_dependencies(
    model_args: "ModelArguments",
) -> None:
    if model_args.infer_backend == "vllm":
        check_version("vllm>=0.4.3,<=0.7.3")
        check_version("vllm", mandatory=True)

def _parse_infer_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    allow_extra_keys = is_env_enabled("ALLOW_EXTRA_ARGS")
    return _parse_args(parser, args, allow_extra_keys=allow_extra_keys)

def get_infer_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _INFER_CLS:
    client_args_list, data_args, generating_args,distill_args,finetuning_args= _parse_infer_args(args)
    return client_args_list, data_args,finetuning_args,generating_args,distill_args

def get_origin_infer_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _ORIGIN_INFER_CLS:
    model_args, data_args, generating_args,distill_args,finetuning_args = _parse_infer_args(args)
    model_args = model_args[0]
    
    if isinstance(model_args,tuple):
        model_args = model_args[0]
    
    return model_args, data_args, finetuning_args,generating_args,distill_args,


def _parse_curation_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _CURATION_CLS:
    parser = HfArgumentParser(_CURATION_ARGS)
    return _parse_args(parser, args)

def get_curation_args(args: Optional[Union[Dict[str, Any], List[str]]] = None) -> _CURATION_CLS:
    curation_args = _parse_curation_args(args)
    return curation_args
    

def _verify_model_args(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args=None,
) -> None:
    pass