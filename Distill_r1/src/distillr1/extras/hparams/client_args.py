import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union, List

import torch
from transformers.training_args import _convert_str_dict
from typing_extensions import Self
from .model_args import ModelArguments
from .generating_args import GeneratingArguments
from .data_args import DataArguments


@dataclass
class ClientArguments(ModelArguments,GeneratingArguments):
    """
    Arguments for client
    """
    