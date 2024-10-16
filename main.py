import argparse
import yaml
from src.model.config import Config
from src.common.message import *
from src.api.api import *
from src.strategy.method import *
from src.strategy.getter import *
from src.tools.tool import *

    
async def main():
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(args.config)         
    api  = API(config)
    text = read_file(config.file_path)
    Method = StrategyGetter.get_strategy(config.method)(api)
    questions,answers = await Method.run(text,config.main_theme,num_question_per_title=10,batch_size=config.batch_size)
    save_QA_dataset(questions,answers,config.save_dir)
    

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    
    
