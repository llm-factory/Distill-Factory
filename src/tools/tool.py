import json
from typing import List, Union
import re
def read_file(file_path:str)-> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def write_json_file(file_path:str, dataset:List[dict]):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
        
        

def clean_and_split_multiline_reply(lines:str)-> List[str]:
    lines = [re.sub(r'^\d+\.', '', l) for l in lines.splitlines()]
    lines = [re.sub(r'^[.、]',"",l) for l in lines]
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 5]
    return lines


def clean_and_split_multiline_replyList(replys:List[str])-> List[str]:
    cleaned =[]
    for lines in replys: 
        lines = [re.sub(r'^\d+\.', '', l) for l in lines.splitlines()]
        lines = [re.sub(r'^[.、]',"",l) for l in lines]
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l) > 5]
        cleaned.extend(lines)
    return cleaned

def save_QA_dataset(questions,answers,save_dir):
    dataset = []
    for q,a in zip(questions,answers):
        dataset.append(
            {"instruction":q,
             "input":"",
             "output":a             
             })
    path = f"{save_dir}/QA.json"
    write_json_file(path,dataset)