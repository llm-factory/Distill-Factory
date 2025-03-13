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
from ..extras.misc import get_device_count
from dataclasses import dataclass

logger = logging.get_logger(__name__)


@dataclass
class ModelInfo:
    model_name_or_path: str
    model_id: str
    allocated_device: List[int]
    model_param: int
    template: str


class ModelRouter:
    """
    Dispatch request.
    """
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        """

        """
        model_args, data_args,finetuning_args,generating_args,distill_args= get_infer_args(args)
        self.model_args = model_args 
        self.data_args = data_args
        
        self.chatmodels: Dict[str, ChatModel] = {}
        self.model_args = model_args or {}
        self.available_model_names = []
        self.models = []
        self.model_infos = []
        self.model_name_or_paths = getattr(model_args, 'model_name_or_path', None)
        self.model_ids = getattr(model_args, 'model_id', None)
        self.templates = getattr(data_args, 'template', None)
        self.init_router()
        # self.model_roles = self.model_args.get("model_role", []).split(",")
        
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
        """
        if not self.model_name_or_paths:
            raise ValueError("model_name_or_paths is required.")
        if not self.model_ids:
            self.model_ids = self.model_name_or_paths.copy() # model_name_or_path as model_id by default
            
        self.model_name_or_paths = self.model_name_or_paths.split(",")
        print(f"model_name_or_paths: {self.model_name_or_paths}")
        print(f"model_ids: {self.model_ids}")
        self.model_ids = self.model_ids.split(",")
        if len(self.model_ids) != len(self.model_name_or_paths):
            raise ValueError("model_ids and model_name_or_paths must have the same number of elements.")
        if not self.templates:
            raise ValueError("templates is required.")
        self.templates = self.templates.split(",")
        
        for i,(model_name_or_path, model_id) in enumerate(zip(self.model_name_or_paths, self.model_ids)):
            self.model_infos.append(ModelInfo(model_name_or_path.strip(),model_id.strip(),allocated_device=[],model_param = None,template=self.templates[i]))
        total_params = 0
        for model in self.model_infos:
            model_param = esitimate_model_params(model.model_name_or_path)
            model.model_param = model_param
            total_params += model_param
        self.model_infos.sort(key=lambda x: x.model_param, reverse=True)
        
        num_devices = get_device_count()
        logger.info_rank0(f"Found {num_devices} CUDA device(s)")
        current_available_device = os.getenv("CUDA_VISIBLE_DEVICES",[i for i in range(num_devices)])
        if isinstance(current_available_device, str):
            current_available_device = [int(i) for i in current_available_device.split(",")]
        logger.info_rank0(f"Available devices: {current_available_device}")

        for i,model_info in enumerate(self.model_infos):
            model_id = model_info.model_id
            
            allocate_num = max(1,min(len(current_available_device),round((model_info.model_param / total_params) * num_devices)))
            model_info.allocated_device = current_available_device[:allocate_num]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in model_info.allocated_device])
            self.chatmodels[model_id] = ChatModel(args=None,model_name_or_path=model_info.model_name_or_path,template=model_info.template) # load model
            logger.info_rank0(f"model {model_id} is allocated to devices {model_info.allocated_device}")
            if i == len(self.model_infos) - 1 and len(current_available_device[allocate_num:]) == 0:
                allocate_num = len(current_available_device)
            else:              
                current_available_device = current_available_device[allocate_num:]

        # self.available_model_names = list(self.models.keys())
        # logger.info_rank0(f"Loaded models: {self.available_model_names}")
        
def esitimate_model_params(model_name_or_path:str)->int:
    """
    Estimate the number of parameters of the model.
    """
    return 7