import asyncio
from typing import List, Optional, Union, Any, Dict
import os
from openai import AsyncOpenAI
from .router import ModelRouter,ModelInfo
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    ChatCompletionMessage,
    ChatCompletionResponseUsage
)


class Client(AsyncOpenAI):
    def __init__(self, base_url: str, api_key: str, **kwargs):
        super().__init__(base_url=base_url, api_key=api_key)
        self.default_model = kwargs.pop("model_id", None)
    
    def _process_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        processed_messages = []
        for message in messages:
            processed_messages.append({"role": "user", "content": message.content})
        return processed_messages

    async def create_chat_completion_response(
            self,
            request: "ChatCompletionRequest",
            **kwargs
    ) -> "ChatCompletionResponse":
        processed_messages = self._process_messages(request.messages)
        model = request.model or self.default_model
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
    
    async def async_chat(self,messages:List[List[Dict[str,str]]],model:str=None)->List[str]:
        """
        """
        response = await asyncio.gather(*(self.create_chat_from_message(message,model) for message in messages))
        return response
    
    async def create_chat_from_message(
            self,
            message: str,
            model: str=None,
            **kwargs
    ) -> "ChatCompletionResponseChoice":
        """
        Create a chat completion from a single message.
        """
        model_to_request = model or self.default_model
        request = ChatCompletionRequest(
            model=model_to_request,
            messages=[
                ChatMessage(role="user", content=message)
            ],
        )
        response = await self.create_chat_completion_response(request, **kwargs)
        return response.choices[0]

    async def judge_answer_correctness(
            self,
            question: str,
            answer: str,
            llm_answer: str
    ) -> bool:
        judge_prompt = f"""
        You are a judge that evaluates the correctness of a solution.
        You are given a question, an answer and a ground truth answer.
        You need to judge whether the answer is correct or not.
        
        Question: {question}
        Answer: {llm_answer}
        Ground Truth Answer: {answer}
        
        Please judge whether the answer is correct or not.
        Please only output \\boxed{{}} in the format of \\boxed{{}}.
        """

        judge_response = await self.create_chat_from_message(judge_prompt)
        if '\\boxed{correct}' in judge_response.message.content:
            return True
        else:
            return False
# TODO

from typing import List, Dict

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
                api_host = model_info.api_host or "localhost"
                api_port = model_info.api_port or "8000"
                base_url = f"http://{api_host}:{api_port}/v1"
            else:
                base_url = model_info.base_url
            client = Client(base_url, model_info.api_key, model_id=model_info.model_id)
            clients[model_info.model_id] = client
            
            if model_info.role and model_info.role not in clients:  # 避免 role 覆盖已有 key
                clients[model_info.role] = client
        else:
            if model_info.base_url:
                client = Client(model_info.base_url, model_info.api_key, model_id=model_info.model_id)
                clients[model_info.model_id] = client
                if model_info.role and model_info.role not in clients:
                    clients[model_info.role] = client
    return clients
