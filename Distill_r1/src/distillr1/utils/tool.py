import json
from typing import List, Union, Dict

def save_to_json(questions, answers, path):
    datas = [[{"instruction":q,"output":a}] for q,a in zip(questions,answers)]
    with open(path, "w",encoding='utf-8') as f:
        json.dump(datas,f,ensure_ascii=False,indent=2)        
        

from typing import List, Union, Dict

def parse_data(datas: List[Union[List[str], Dict[str, str],List[Dict[str,str]], str]]) -> List[str]:
    parsed_datas = []
    
    for data in datas:
        if isinstance(data, list):
            parsed_datas.extend(data)
        elif isinstance(data, dict):  # 处理字典
            parsed_datas.extend(data.values())
        elif isinstance(data, str):  # 处理字符串
            parsed_datas.append(data)
    
    return parsed_datas
    
    for data in datas:
        if isinstance(data, dict):  # 处理字典格式的数据
            if "content" in data and "问题：" in data["content"]:
                question_part = data["content"].split("问题：", 1)[-1]  # 提取问题部分
                questions.append(question_part)
        elif isinstance(data, list):  # 处理列表格式的数据
            for item in data:
                if isinstance(item, str) and "问题：" in item:
                    question_part = item.split("问题：", 1)[-1]
                    questions.append(question_part)
        elif isinstance(data, str) and "问题：" in data:  # 处理单独的字符串
            question_part = data.split("问题：", 1)[-1]
            questions.append(question_part)

    return questions