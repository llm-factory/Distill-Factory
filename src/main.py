import argparse
import asyncio

from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from tools.tool import read_file, save_QA_dataset
from log.logger import setup_logger
    
async def main():
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(file_path=args.config)         
    api  = API(config)
    logger = setup_logger("output.log")
    Method = StrategyGetter.get_strategy(config.method)(api)
    questions,answers = await Method.run(config,num_question_per_title=10,concurrent_api_requests_num=config.concurrent_api_requests_num)
    save_QA_dataset(questions,answers,config.save_dir,config.save_file_name)
    

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    
    
