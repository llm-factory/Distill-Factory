import json
from typing import List, Dict
from openai import OpenAI as OpenAIClient
from openai import AsyncOpenAI
import asyncio
from model.config import Config
from common.message import BaseMessage

class BaseOpenAI():
    def __init__(self, _config):
        self._model = _config.model
        self.temperature = _config.temperature
        self.base_url = _config.base_url
        self.api_key = _config.api_key
    @staticmethod
    def _parse_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
        return [{"role": message["role"], "content": message["content"]} for message in messages]


class API(BaseOpenAI):
    def __init__(self,_config:Config):
        super().__init__(_config)
        self.client = OpenAIClient(
            max_retries=5,
            timeout=120.0,
            base_url=self.base_url,
            api_key=self.api_key
        )
        self.async_client = AsyncOpenAI(
            max_retries=5,
            timeout=120.0,
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_api_reply(self,messages: List[Dict[str, str]], **kwargs) -> str:
        result = self.client.chat.completions.create(messages=messages, 
                                                        model=self._model,
                                                        **kwargs)        
        return result.choices[0].message.content    

    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        response = self.get_api_reply(messages, **kwargs)
        return response
    
    async def async_get_api_reply(self,messages:List[BaseMessage], **kwargs) -> str:
        result = await self.async_client.chat.completions.create(
            messages=messages,
            model=self._model,
            **kwargs 
        )
        return result.choices[0].message.content
    
    async def async_chat(self, messages: List[List[BaseMessage]], **kwargs) -> List[str]:
        response= await asyncio.gather(*(self.async_get_api_reply(message,**kwargs) for message in messages))
        return response
