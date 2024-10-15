from typing import Dict
from pathlib import Path

import yaml


class Config:
    def __init__(self, filename):
        self.conf_dict = yaml.safe_load(open(filename))
        openai_config: Dict = self.conf_dict.get("openai")
        self.openai_base_url: str = openai_config.get("base_url")
        self.openai_api_key: str = openai_config.get("api_key")
        self.openai_model: str = openai_config.get("model")
        self.temperature: float = openai_config.get("temperature")


root_path = Path(__file__).parent.parent
config = Config(root_path / "../config.yml")
