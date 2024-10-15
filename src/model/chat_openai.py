import json
from typing import List, Dict

from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, wait_random_exponential, stop_after_attempt

from config import config
from src.common.message import BaseMessage


class ChatOpenAI:
    def __init__(self):
        self._client = OpenAI(
            max_retries=5,
            timeout=30.0,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key
        )
        self._model = config.openai_model

    @staticmethod
    def _parse_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        return [{"role": message["role"], "content": message["content"]} for message in messages]

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _completion_with_backoff(self, messages: List[BaseMessage], **kwargs) -> ChatCompletion:
        request_kwargs = {
            "messages": self._parse_messages(messages),
            "model": self._model,
            "temperature": config.temperature,
        }
        return self._client.chat.completions.create(**request_kwargs, **kwargs)

    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        response = self._completion_with_backoff(messages=messages, **kwargs).choices[0].message.content
        return response
