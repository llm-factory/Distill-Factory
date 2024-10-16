import argparse
import yaml
from src.model.config import Config
from src.common.message import *
from src.api.api import *
from src.strategy.genQA import *


    
async def main():
    parser = argparse.ArgumentParser(description="载入配置文件")
    parser.add_argument("config", type=str, help="filename of config")
    args = parser.parse_args()
    config = Config(args.config)         
    api  = API(config)
    text = read_file(config.file_path)
    genMethod = genQA(api)
    questions,answers = await genMethod.gen(text,"巴黎奥运会",num_question_per_title=10,batch_size = 1)
    save_QA_dataset(questions,answers,config.save_dir)
    

if __name__ == "__main__":
    asyncio.run(main())
    
    
    
    
    
    
    
    
