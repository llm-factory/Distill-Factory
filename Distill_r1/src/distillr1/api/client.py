import asyncio
from typing import List, Optional, Union, Any, Dict
import os
from openai import AsyncOpenAI
from .router import ModelRouter,ModelInfo
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args
from .misc import convert_api_compatable_generating_args
from typing import List, Dict
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionMessage,
    ChatCompletionResponseUsage,
    UserMessage
)


class Client(AsyncOpenAI):
    def __init__(self, base_url: str, api_key: str, **kwargs):
        super().__init__(base_url=base_url, api_key=api_key)
        self.base_url = base_url
        self.api_key = api_key
        
        self.model_info = kwargs.pop("model_info", None)
        print("model_info")
        print(self.model_info)
        self.model_id = self.model_info.model_id if self.model_info else None
        self.generating_args = self.model_info.get_generating_args() if self.model_info else None
    
    def _process_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        processed_messages = []
        for message in messages:
            processed_messages.append({"role": message.role.value, "content": message.content})
        return processed_messages

    def get_generating_args(self) -> Dict:
        return self.generating_args

    async def create_chat_completion_response(
            self,
            request: "ChatCompletionRequest",
            **kwargs
    ) -> "ChatCompletionResponse":
        print("request")
        print(request.messages)
        
        processed_messages = self._process_messages(request.messages)
        model = request.model or self.default_model
        print("model")
        print(model)
        print("api_key")
        print(self.api_key)
        print("processed_messages")
        print(processed_messages)
        
        response = await self.chat.completions.create(
            model=model,
            messages=processed_messages,
            **kwargs
        )
        responses = []
        for i in response.choices:
            reasoning_content = getattr(i.message, 'reasoning_content', None) or getattr(i.message, 'reasoning', None) or None
            finish_reason = i.finish_reason if i.finish_reason in {'stop', 'length', 'tool_calls'} else 'stop'
            responses.append(
                ChatCompletionResponseChoice(
                    index=i.index,
                    message=ChatCompletionMessage(
                        role=i.message.role,
                        content=i.message.content,
                        reasoning_content=reasoning_content
                    ),
                    finish_reason=finish_reason
                )
            )

        return ChatCompletionResponse(
            id=model,
            model=model,
            choices=responses,
            usage=ChatCompletionResponseUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        )
    
    async def async_chat(self,messages:Union[List[List[ChatMessage]],List[str]],model:str=None,**kwargs)->List[str]:
        """
        """
        print("async chat messages:")
        print(messages)
        
        responses = await asyncio.gather(*(self.create_chat_from_message(message,model,**kwargs) for message in messages))
        responses_str = [response.message.content for response in responses]
        return responses_str
    
    async def create_chat_from_message(
            self,
            message: List[ChatMessage],
            model: str=None,
            **kwargs
    ) -> "ChatCompletionResponseChoice":
        """
        Create a chat completion from a single message.
        """
        # 如果 kwargs 提供了 generating_args ,则覆盖.
        # 否则使用 model_info.get_genearting_args()
        generating_args = convert_api_compatable_generating_args(self.model_info.get_generating_args())
        for key, value in kwargs.items():
            if key in generating_args:
                generating_args[key] = value        
        model_to_request = model or self.model_info.model_id        
        request = ChatCompletionRequest(
            model=model_to_request,
            messages=message,
        )
        response = await self.create_chat_completion_response(request, **generating_args)
        return response.choices[0]

    async def judge_answer_correctness(
            self,
            llm_answer: str,
            answer: str
    ) -> bool:
        judge_prompt = f"""
You are a judge that evaluates the correctness of a solution.
You are given an solution and a ground truth answer.
You need to first extract answers from the solution, then judge whether the answer is correct or not compared with the ground truth.
Solution: {llm_answer}
Ground Truth Answer: {answer}

Please judge whether the Solution is correct or not.
Output \\boxed{{correct}} or \\boxed{{incorrect}} only.
"""

        judge_response = await self.create_chat_from_message([UserMessage(judge_prompt)])
        print("JUDGE RESPONSE")
        print(judge_response)
        if '\\boxed{correct}' in judge_response.message.content:
            return True
        else:
            return False
# TODO

def parse_client(model_infos: List[ModelInfo]) -> Dict[str, Client]:
    """
    Return a dictionary of Client instances, indexed by model_id and role.

    Args:
        model_infos (List[ModelInfo]): List of model information objects.

    Returns:
        Dict[str, Client]: A dictionary containing model clients.
    """
    clients = {}
    for model_info in model_infos:
        if model_info.base_url and model_info.base_url in clients:
            continue

        if model_info.deploy:
            if model_info.base_url is None:  # 处理 base_url 为空的情况
                api_host = model_info.api_host or os.getenv("API_HOST","0.0.0.0")
                api_port = model_info.api_port or int(os.getenv("API_PORT",8000))
                base_url = f"http://{api_host}:{api_port}/v1"
            else:
                base_url = model_info.base_url
            client = Client(base_url, model_info.api_key, model_info=model_info)
            clients[model_info.model_id] = client
            
            if model_info.role and model_info.role not in clients:  # 避免 role 覆盖已有 key
                clients[model_info.role] = client
        else:
            if model_info.base_url:
                client = Client(model_info.base_url, model_info.api_key, model_info=model_info)
                clients[model_info.model_id] = client
                if model_info.role and model_info.role not in clients:
                    clients[model_info.role] = client
    return clients
