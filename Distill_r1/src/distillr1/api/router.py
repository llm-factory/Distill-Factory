# Copyright 2025 the LlamaFactory team.
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

import os
import json
from typing import Dict, List, Optional, Any
import copy
from ..hparams import get_infer_args
from ..chat import ChatModel
from ..extras import logging
from .protocol import ModelInfo
from ..extras.misc import get_device_count
from dataclasses import dataclass
from enum import Enum
logger = logging.get_logger(__name__)




class ModelRouter:
    """
    Dispatch request.
    """
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        """
        """
        chatmodel_args_list, data_args,finetuning_args,generating_args,distill_args = get_infer_args(args)
        self.chatmodel_arg_list = chatmodel_args_list
        self.data_args = data_args
        self.model_infos = []
        self.chatmodels = {}
        self.deploy = False
        self.init_router()

    def get_deploy(self) -> bool:
        return self.deploy

    def get_model_infos(self) -> List[ModelInfo]:
        """
        return : list of model infos
        """
        return self.model_infos
        
    def get_model_ids(self)-> List[str]:
        """
        return : list of model ids
        """
        return [model.model_id for model in self.model_infos]
    def get_model(self, model_id: str) -> ChatModel:
        """
        Get the model by model_id.
        """
        return self.chatmodels[model_id]

    def init_router(self):
        """
        check and Load models.
        Parse model_infos here
        """

        for i, (model_args, generating_args,data_args) in enumerate(self.chatmodel_arg_list):
            
            model_config_dict = {
                "model_name_or_path": model_args.model_name_or_path.strip(),
                "model_id": model_args.model_id or model_args.model_name_or_path.strip(),
                "device": model_args.device,
                "role": model_args.role,
                "deploy": model_args.deploy,
                "base_url": model_args.base_url,
                "api_key": model_args.api_key,
                "temperature": generating_args.temperature,
                "top_p": generating_args.top_p,
                "top_k": generating_args.top_k,
                "num_beams": generating_args.num_beams,
                "max_length": generating_args.max_length,
                "max_new_tokens": generating_args.max_new_tokens,
                "repetition_penalty": generating_args.repetition_penalty,
                "length_penalty": generating_args.length_penalty,
                "default_system": generating_args.default_system,
                "skip_special_tokens": generating_args.skip_special_tokens,
                "template": data_args.template,
            }
            model_info = ModelInfo(**model_config_dict)
            if model_info.deploy or model_info.device:
                os.environ["CUDA_VISIBLE_DEVICES"] = model_info.device
                print(model_info.device)
                self.chatmodels[model_info.model_id] = ChatModel(
                    args=model_config_dict
                )
                logger.info_rank0(f"model {model_info.model_id} is allocated to devices {model_info.device}")
            else:
                logger.info_rank0(f"model {model_info.model_id} don't need to be deployed locally")
            self.model_infos.append(model_info)