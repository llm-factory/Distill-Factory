import json
from typing import List, Dict,Union
from openai import OpenAI as OpenAIClient
from openai import AsyncOpenAI
import asyncio
from model.config import Config
from common.message import BaseMessage,buildMessages,UserMessage

class BaseOpenAI:
    def __init__(self, _config: Union[Config, Dict]):
        if isinstance(_config, Config):
            self._model = _config.model
            self.temperature = _config.temperature
            self.base_url = _config.base_url
            self.api_key = _config.api_key
        elif isinstance(_config, Dict):
            self._model = _config.get("model")
            self.temperature = _config.get("temperature", 1.0)
            self.base_url = _config.get("base_url")
            self.api_key = _config.get("api_key")
            
            
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

    def get_api_reply(self, messages: List[Dict[str, str]], retrieve=False, **kwargs) -> str:
        if retrieve:
            query = ""
            for message in messages:
                if message["role"] == "user":
                    query = message["content"]
                    break
                        
            body = {
                "query": query,
                **kwargs
            }
            print("body:",body)
            result = self.client.post(
                path="/retrieve",
                cast_to=object,
                body=body
            )
            print(result)
            return result["choices"][0]["message"]["content"]
        else:
            result = self.client.chat.completions.create(
                messages=messages,
                model=self._model,
                **kwargs
            )
            return result.choices[0].message.content

    def chat(self, messages: Union[List[BaseMessage], str], retrieve=False, **kwargs) -> str:
        if isinstance(messages, str):
            messages = buildMessages(UserMessage(messages))
        response = self.get_api_reply(messages, retrieve=retrieve, **kwargs)
        return response
    
    async def async_get_api_reply(self,messages:List[BaseMessage], retrieve = False,**kwargs) -> str:
        if retrieve:
            print(retrieve)
            query = ""
            for message in messages:
                if message["role"] == "user":
                    query = message["content"]
                    break                        
            body = {
                "query": query,
                **kwargs
            }
            result = await self.async_client.post(
                path="/retrieve",
                cast_to=object,
                body=body
            )
            return result["choices"][0]["message"]["content"]
            
        else:    
            result = await self.async_client.chat.completions.create(
                messages=messages,
                model=self._model,
                **kwargs 
            )
            return result.choices[0].message.content
        
    async def async_chat(self, messages: Union[List[List[BaseMessage]],List[str]],retrieve=False, **kwargs) -> List[str]:
        if len(messages) and isinstance(messages[0], str):
            messages = [buildMessages(UserMessage(message)) for message in messages]
        response= await asyncio.gather(*(self.async_get_api_reply(message,retrieve,**kwargs) for message in messages))
        return response
