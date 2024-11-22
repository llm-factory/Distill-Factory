from typing import Dict, Optional, Union
from pathlib import Path
import yaml

class Config:
    def __init__(self, config_dict):
        if config_dict:
            self.conf_dict = config_dict

        print(self.conf_dict.get("path"))
        openai_config: Dict = self.conf_dict.get("openai", {})
        self.base_url: str = openai_config.get("base_url")
        if self.base_url is None:
            raise ValueError("base_url is required")
        self.api_key: str = openai_config.get("api_key")
        self.model: str = openai_config.get("model")
        self.temperature: float = openai_config.get("temperature", 1.0)
        self.save_dir: str = self.conf_dict.get("save_dir", "./")
        self.main_theme: str = self.conf_dict.get("main_theme")
        self.file_path: str = self.conf_dict.get("file_path",None)
        self.file_folder: str = self.conf_dict.get("file_folder")
        self.concurrent_api_requests_num: int = self.conf_dict.get("concurrent_api_requests_num", 1)
        self.method: str = self.conf_dict.get("method")