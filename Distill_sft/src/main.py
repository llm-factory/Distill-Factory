import argparse
import asyncio

from model.config import Config
from api.api import API
from strategy.getter import StrategyGetter
from log.logger import Logger
    
async def run_exp():
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(file_path=args.config)         
    api  = API(config)
    logger = Logger()
    Method = StrategyGetter.get_strategy(config.method)(api,config)
    await Method.run(config)
    

if __name__ == "__main__":
    asyncio.run(run_exp())
    
    
    
    
    
    
    
    
