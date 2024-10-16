import argparse
import yaml
from src.model.config import Config
from src.common.message import *
from src.api.api import *
from src.strategy.genQA import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(args.config)         
    print(config.api_key)
    
    nms = buildMessages(
        [SystemMessage(content = "你是一个助手"), 
         UserMessage(content = "请问你是谁开发的？你的名字是什么")
         ])
    
    api  = API(config)
    print(api.chat(nms))
    
    
    
    
    
    
    
    
