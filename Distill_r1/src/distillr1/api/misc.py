from typing import Any,List,Dict,Union
from dataclasses import dataclass
from .protocol import ChatMessage,ModelInfo
from typing import Dict

def try_api_call(args:Any):
    pass


from typing import List

def parse_model_info(model_args) -> List[ModelInfo]:
    model_name_or_paths = getattr(model_args, 'model_name_or_path', None)
    model_ids = getattr(model_args, 'model_id', None)
    roles = getattr(model_args, 'roles', ["chat", "reward"])
    deploy = getattr(model_args, 'deploy', [False, False])
    templates = getattr(model_args, 'templates', None)
    
    if model_name_or_paths is None:
        raise ValueError("model_name_or_path are required.")
    
    model_name_or_paths = model_name_or_paths.split(",")
    if model_ids is None:
        model_ids = model_name_or_paths.copy() # model_name_or_path as model_id by default
    model_ids = model_ids.split(",")
    if True in deploy and templates is None:
        raise ValueError("templates are required for deploy model locally.")
    templates = templates.split(",")
    if len(templates) != len(model_name_or_paths):
        raise ValueError("model_name_or_paths and templates must have the same number of elements. Please fill in the template for models that need to be deployed locally in the corresponding position. Use 'None' to indicate no local deployment, separated by ',' ")
    
    if len(model_ids) != len(model_name_or_paths):
        raise ValueError("model_ids and model_name_or_paths must have the same number of elements.")
        
    model_infos = []
    for i, (model_name_or_path, model_id) in enumerate(zip(model_name_or_paths, model_ids)):
        model_infos.append(ModelInfo(
            model_name_or_path.strip(),
            model_id.strip(),
            allocated_device=[],
            model_param=None,
            template=templates[i]
        ))
    
    return model_infos


def build_messages(*messages: Union[ChatMessage]) -> List[Dict[str,str]]:
    newMessages = []
    for message in messages:
        msg = {"role": message.role, "content": message.content}
        newMessages.append(msg)
    return newMessages

def convert_api_compatable_generating_args(generating_args: Dict) -> Dict:
    api_args = {}
    if "do_sample" in generating_args:
        api_args["temperature"] = generating_args["temperature"] if generating_args["do_sample"] else 0.0
    if "top_p" in generating_args:
        api_args["top_p"] = generating_args["top_p"]
    elif "max_length" in generating_args:
        api_args["max_tokens"] = generating_args["max_length"]
    elif "tool" in generating_args:
        api_args["tool"] = generating_args["tool"]
    elif "n" in generating_args:
        api_args["n"] = generating_args["n"]
            
    return api_args


