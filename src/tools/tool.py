import json
from typing import List, Union
import re
import os
from pathlib import Path
import logging
logger = logging.getLogger("logger")
def read_file(file_path:str)-> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def write_json_file(file_path:str, dataset:List[dict]):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
        
def clean_and_split_reply(lines:str)-> List[str]:
    lines = lines.split('\n')
    lines = [l.strip(".#`\\\"\' ") for l in lines]
    lines = [re.sub(r'^[ :：\.、,\'\"]?\d+[ :：\.、]', '', l) for l in lines]
    lines = [l for l in lines if len(l) > 10]
    return lines

def clean_and_split_reply_list(replys:List[str])-> List[str]:
    cleaned =[]
    for reply in replys: 
        reply = clean_and_split_reply(reply)
        cleaned.extend(reply)
    return cleaned

def clean_and_split_titles(titles:str)-> List[str]:
    titles = titles.split('\n')
    titles = [re.sub(r'^[ :：\.、]?(\d+)[ :：\.、]', '', l) for l in titles]
    titles = [re.sub(r'^([小]?标题)[ :：\.、\d]?', '', l) for l in titles]
    titles = [l.strip() for l in titles]
    titles = [l for l in titles if len(l) >= 3]
    return titles

def clean_and_split_title_list(titlelist:List[str])->List[str]:
    cleaned =[]
    for title in titlelist:
        title = clean_and_split_titles(title)        
        cleaned.extend(title)
    return cleaned
    
    
def save_QA_dataset(questions,answers,save_dir,name):
    dataset = []
    for q,a in zip(questions,answers):
        dataset.append(
            {"instruction":q,
             "input":"",
             "output":a             
             })
    path = f"{save_dir}/{name}"
    write_json_file(path,dataset)
    print(f"dataset with {len(dataset)} samples saved to {path}")
    
def getFilePaths(folder,file,file_type:list[str]):
    paths = []
    if(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.split('.')[-1] in file_type:
                    paths.append(Path(root,file))
    else:
        paths.append(file)
    return paths

def parseJsons(text: List[str],key:str) -> List[str]:
    originalLength = len(text)
    Items = []
    for t in text:
        try:
            parsed_json = json.loads(t)
        except json.JSONDecodeError as e:
            logger.error(f"Error: {e}")
            continue
        for q in parsed_json:
            item = q.get(key)
            if item is not None:
                Items.append(item)
            else:
                logger.error(f"Warning: '{key}' key missing in {q}")
    parsedLength = len(Items)
    
    logger.warning(f"ratio: {parsedLength}/{originalLength}")
    
    return Items