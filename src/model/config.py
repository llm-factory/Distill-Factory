from typing import Dict
from pathlib import Path

import yaml


class Config:
    def __init__(self, filename):
        self.conf_dict = yaml.safe_load(open(filename))
        openai_config: Dict = self.conf_dict.get("openai")
        self.base_url: str = openai_config.get("base_url")
        self.api_key: str = openai_config.get("api_key")
        self.model: str = openai_config.get("model")
        self.temperature: float = openai_config.get("temperature","1.0")
        self.save_dir: str = self.conf_dict.get("save_dir")
        self.main_theme: str= self.conf_dict.get("main_theme")
        self.file_path: Path = self.conf_dict.get("file_path")
        self.batch_size: int = self.conf_dict.get("batch_size")
        self.method : str = self.conf_dict.get("method")