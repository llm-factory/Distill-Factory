import argparse
import asyncio

from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from tools.tool import read_file, save_QA_dataset

    
async def main():
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(args.config)         
    api  = API(config)
    Method = StrategyGetter.get_strategy(config.method)(api)
    questions,answers = await Method.run(config,num_question_per_title=10,concurrent_api_requests_num=config.concurrent_api_requests_num)
    save_QA_dataset(questions,answers,config.save_dir,"test.json")
    

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    
    
