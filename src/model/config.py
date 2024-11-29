from typing import Dict, Optional, Union
from pathlib import Path
import yaml

class Config:
    def __init__(self, config_dict=None,file_path=None):
        if config_dict:
            self.conf_dict = config_dict
        if file_path:
            self.file_path = file_path
            self.conf_dict = self.load_config(file_path)
        openai_config: Dict = self.conf_dict.get("openai", {})
        self.base_url: str = openai_config.get("base_url")
        if self.base_url is None:
            raise ValueError("base_url is required")
        self.api_key: str = openai_config.get("api_key")
        if self.api_key is None:
            raise ValueError("api_key is required")
        self.model: str = openai_config.get("model")
        if self.model is None:
            raise ValueError("model is required")
        self.temperature: float = openai_config.get("temperature", 1.0)
        self.save_dir: str = self.conf_dict.get("save_dir", "./")
        self.main_theme: str = self.conf_dict.get("main_theme")
        self.file_path: str = self.conf_dict.get("file_path",None)
        self.file_folder: str = self.conf_dict.get("file_folder")
        self.concurrent_api_requests_num: int = self.conf_dict.get("concurrent_api_requests_num", 1)
        self.method: str = self.conf_dict.get("method")
        self.file_type: str = self.conf_dict.get("file_type", "txt")
        self.save_file_name: str = self.conf_dict.get("save_file_name", "dataset.json")
        self.is_structure_data: bool = self.conf_dict.get("is_structure_data", False)
        self.text_template: str = self.conf_dict.get("text_template", None)
    
    def load_config(self, file_path: str) -> Dict:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)