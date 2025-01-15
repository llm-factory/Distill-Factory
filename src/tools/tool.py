from typing import List, Union, Dict, Optional
import re
import os
from pathlib import Path
import logging
import json
from model.config import *

logger = logging.getLogger("logger")

def read_file(file_path:str)-> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_datas(file_path: Path,config: Dict) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        if config.is_structure_data and file_path.suffix == '.json':
            datas = json.load(file) # list of dict
            formatted_texts = []
            for data in datas:
                text = format_structured_data(data, config.text_template)
                texts = [text[i:i+config.chunk_size] for i in range(0, len(text)-config.chunk_size//10, config.chunk_size)]
                formatted_texts.extend(texts)
            return formatted_texts
        elif config.is_structure_data and file_path.suffix != '.json':
            raise Exception("is_structure_data is True but file is not json")
        else: # is_structure_data is False
            if file_path.suffix in ['.rst','.txt','.md','.html']: # read directly
                texts = file.read()
                textsList = [texts[i:i+config.chunk_size] for i in range(0, len(texts)-config.chunk_size//10, config.chunk_size)]
                return textsList
            else:
                pass # TODO
# def parse_data(raw_data, config):
#     """
#     raw_data: str or dict
#     return: list[str]
#     """
#     if not config.is_structure_data:
#         return [raw_data]

#     try:
#         structured_data = load_json(raw_data)
#         if not structured_data:
#             return [raw_data]
            
#         formatted_text = format_structured_data(
#             structured_data,
#             config.text_template
#         )
#         return [formatted_text]
        
#     except Exception as e:
#         raise Exception(f"Error: {e}")

def write_json_file(file_path:str, dataset:List[dict]):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)
        
def format_structured_data(data: Dict, template: str) -> str:
    try:
        formatted_data = {}
        logger.debug(f"template: {template}")
        for key, value in data.items():
            logger.debug(f"key: {key}, value: {value}")
            
            if isinstance(value, list):
                formatted_data[key] = '、'.join(str(v) for v in value)
            else:
                formatted_data[key] = value            
        return template.format(**formatted_data)
    except KeyError as e:
        raise KeyError(f"Template missing key: {str(e)}")
    except Exception as e:
        raise Exception(f"Error: {e}")

def load_json(data: Union[str, Dict]) -> Dict:
    if isinstance(data, dict):
        return data        
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}        

def clean_and_split_reply(lines:str)-> List[str]:
    lines = lines.split('\n')
    lines = [l.strip("。.^#`\\\"\' ") for l in lines]
    lines = [re.sub(r'^[ :：\.、,\'\"]?\d+[ :：\.、]', '', l) for l in lines]
    lines = [l.strip("。.^#`\\\"\' ") for l in lines]
    lines = [l for l in lines if len(l) > 10]
    return lines

def clean_and_split_reply_list(replies:List[str])-> List[str]:
    cleaned =[]
    for reply in replies: 
        reply = clean_and_split_reply(reply)
        cleaned.extend(reply)
    return cleaned

def clean_and_split_question_list(replies:List[str])-> List[str]:
    cleaned = []
    pattern = r'问题[\d]+::(.*?)(?=问题[\d]+::|$)'
    for reply in replies:
        matches = re.finditer(pattern, reply, re.DOTALL)
        matches_list = list(matches)
        questions = []
        for match in matches_list:
            question = match.group(1).strip()
            if question:
                questions.append(question)
        cleaned.extend(questions)
        
    return cleaned
        

def clean_and_split_titles(titles:str)-> List[str]:
    titles = titles.split('\n')
    titles = [re.sub(r'^[ :：\.、]?(\d+)[ :：\.、]', '', l) for l in titles]
    titles = [re.sub(r'^([小]?标题)[ :：\.、\d]?', '', l) for l in titles]
    titles = [l.strip() for l in titles]
    titles = [l for l in titles if len(l) >=3]
    return titles

def clean_and_split_title_list(titlelist:List[str])->List[str]:
    cleaned =[]
    for title in titlelist:
        title = clean_and_split_titles(title)        
        cleaned.extend(title)
    return cleaned
    

def init_QA_dataset(save_dir,name):
    datas = []
    path = f"{save_dir}/{name}"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(datas, file, ensure_ascii=False, indent=2)
        

def save_QA_dataset(questions,answers,save_dir,name,max_nums): 
    path = f"{save_dir}/{name}"
    with open(path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
    
    dataset = []
    for q,a in zip(questions,answers):
        dataset.append(
            {"instruction":q,
             "input":"",
             "output":a             
             })
    datas.extend(dataset)
    if len(datas) > max_nums:
        datas = datas[:max_nums]
        logger.warning(f"max_nums set to {max_nums}, dataset is truncated")
    write_json_file(path,datas)
    logger.info(f"total samples num of {path}: {len(datas)}")
    
def getFilePaths(config:Config)->List[Path]:
    folder = config.file_folder
    file = config.file_path
    file_type = config.file_type
    paths = []
    # print("file:",file)
    # print("folder",folder)
    if(folder):
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.split('.')[-1] in file_type:
                    paths.append(Path(root,file))
    else:
        paths.extend(file)
    return paths

def parse_response(response: str, regex_splitter: str = "问题[\d ]+::") -> List[str]:
    """
    parse response given regex splitter
    """
    responses = []    
    pattern = rf'(?:{regex_splitter})(.*?)(?={regex_splitter}|$)'
    matches = re.finditer(pattern, response, re.DOTALL)
    matches_list = list(matches)
    for match in matches_list:
        response = match.group(1).strip("")
        response = response.lstrip(". ")
        if response:
            responses.append(response)
    return responses

def parse_responses(responses:List[str], splitter: str = "问题[\d ]+::")->List[str]:
    cleaned = []
    for response in responses:
        cleaned.extend(parse_response(response,splitter))
    return cleaned


def clean_and_split_question_list(replies:List[str])-> List[str]:
    cleaned = []
    pattern = r'问题[\d]+::(.*?)(?=问题[\d]+::|$)'
    for reply in replies:
        matches = re.finditer(pattern, reply, re.DOTALL)
        matches_list = list(matches)
        questions = []
        for match in matches_list:
            question = match.group(1).strip()
            if question:
                questions.append(question)
        cleaned.extend(questions)
        
    return cleaned
        