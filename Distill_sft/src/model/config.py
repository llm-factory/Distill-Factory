from typing import Dict, Optional, Union,List
from pathlib import Path
import yaml
from dataclasses import dataclass
import logging

logger = logging.getLogger('logger')


@dataclass
class APIConfig:
    base_url: str = None
    api_key: str = None
    model: str = None
    temperature: float = 1.0

@dataclass
class FileConfig:
    file_path: List[str]
    file_folder: Optional[str] = None
    is_structure_data: bool = False
    text_template: Optional[str] = None
    chunk_size: int = 2048
    file_type: List[str] = None
    main_theme: str = ""


@dataclass
class GenerationConfig:
    method: str
    num_questions_per_title: int = 5
    splitter: str = "问题::"
    concurrent_requests: int = 1
    question_prompt: Optional[str] = None
    answer_prompt: Optional[str] = None    
    save_file_name: str = "dataset.json"
    save_dir: str = "./"
    concurrent_api_requests_num: int = 1
    max_nums: int = 1e6
    quantity_level: int = 3
    diversity_mode: str = "basic"
    verify_qa: bool = False

@dataclass
class RagConfig:
    enable_rag: bool = False
    rag_api_config: APIConfig = None
class Config:
    def __init__(self, config_dict=None,file_path=None):
        if config_dict:
            self.conf_dict = config_dict
        if file_path:
            self.file_path = file_path
            self.conf_dict = self.load_config(file_path)
        
        
        api_config_dict = self.conf_dict.get("api", {})
        self.api_config = APIConfig(
            base_url=api_config_dict.get("base_url"),
            api_key=api_config_dict.get("api_key"),
            model=api_config_dict.get("model"),
            temperature=api_config_dict.get("temperature", 1.0)
        )
        self.base_url = self.api_config.base_url
        self.api_key = self.api_config.api_key
        self.model = self.api_config.model
        self.temperature = self.api_config.temperature        
        if self.base_url is None:
            raise ValueError("base_url of api is required")
        if self.api_key is None:
            raise ValueError("api_key of api is required")
        self.model: str = self.api_config.model
        if self.model is None:
            raise ValueError("model name of api is required")
        
        
        
        file_config_dict = self.conf_dict.get("file", {})        
        if not file_config_dict:
            raise ValueError("file config is required")

        file_type = file_config_dict.get("file_type", None)

        if not file_type:
            if self.conf_dict.get("generation", {}).get("method") == "VisGen":
                file_type = "pdf"
            else:
                file_type = "txt"

        if isinstance(file_type, str):
            file_type = [file_type]

        self.file_config = FileConfig(
            file_path=file_config_dict.get("file_path","").split(),
            file_folder=file_config_dict.get("file_folder"),
            main_theme=file_config_dict.get("main_theme",""),
            is_structure_data=file_config_dict.get("is_structure_data", False),
            text_template=file_config_dict.get("text_template"),
            chunk_size=file_config_dict.get("chunk_size", 2048),
            file_type=file_type
        )
        self.main_theme: str = self.file_config.main_theme
        self.file_path: List[str] = self.file_config.file_path
        self.file_folder: str = self.file_config.file_folder
        self.is_structure_data: bool = self.file_config.is_structure_data
        self.text_template: str = self.file_config.text_template        
        self.chunk_size: int = self.file_config.chunk_size
        self.file_type: List[str] = self.file_config.file_type
        if not self.file_path and not self.file_folder:
            raise ValueError("file_path or file_folder is required")
        if self.file_folder and not self.file_type:
            raise ValueError("file_type is required if file_folder is True")
        
        
        
        generation_config_dict = self.conf_dict.get("generation", {})
        self.generation_config = GenerationConfig(
            method=generation_config_dict.get("method"),
            num_questions_per_title=generation_config_dict.get("num_questions_per_title", 5),
            splitter=generation_config_dict.get("splitter", None),
            concurrent_requests=generation_config_dict.get("concurrent_requests", 1),
            question_prompt=generation_config_dict.get("question_prompt"),
            answer_prompt=generation_config_dict.get("answer_prompt"),
            save_file_name=generation_config_dict.get("save_file_name", None),
            save_dir=generation_config_dict.get("save_dir", "./"),
            concurrent_api_requests_num=generation_config_dict.get("concurrent_api_requests_num", 1),
            max_nums=generation_config_dict.get("max_nums", 1e6),
            quantity_level = generation_config_dict.get("quantity_level", 3),
            diversity_mode = generation_config_dict.get("diversity_mode", "basic"),
            verify_qa= generation_config_dict.get("verify_qa", False)
        )
        self.method: str = self.generation_config.method
        self.num_questions_per_title: int = self.generation_config.num_questions_per_title
        self.splitter: str = self.generation_config.splitter
        self.concurrent_requests: int = self.generation_config.concurrent_requests
        self.question_prompt: str = self.generation_config.question_prompt
        self.answer_prompt: str = self.generation_config.answer_prompt
        self.save_file_name: str = self.generation_config.save_file_name
        self.save_dir: str = self.generation_config.save_dir
        self.concurrent_api_requests_num: int = self.generation_config.concurrent_api_requests_num
        self.max_nums: int = self.generation_config.max_nums
        self.quantity_level: int = self.generation_config.quantity_level
        self.diversity_mode: str = self.generation_config.diversity_mode
        self.verify_qa: bool = self.generation_config.verify_qa
        
        if not self.method:
            raise ValueError("generation method is required")
        if not self.save_file_name:
            raise ValueError("save_file_name is required")
    
    
        
        rag_conf_dict = self.conf_dict.get("rag", {})
        self.enable_rag = rag_conf_dict.get("enable_rag", False)
        self.rag_api_config = rag_conf_dict.get("api", None)
        if self.rag_api_config:
            self.rag_api_key = self.rag_api_config.get("api_key", None)
            self.rag_model_name = self.rag_api_config.get("model", None)
            self.rag_temperature = self.rag_api_config.get("temperature", 1.0)
            self.rag_base_url = self.rag_api_config.get("base_url", None)
            self.rag_config = RagConfig(
                enable_rag=self.enable_rag,
                rag_api_config=self.rag_api_config
            )
        
        if self.enable_rag and not self.rag_api_config:
            raise ValueError("rag api config is required if enable rag")
        
        
    def load_config(self, file_path):
        with open(file_path, "r",encoding='utf-8') as f:
            return yaml.safe_load(f)