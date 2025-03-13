import asyncio
from typing import List, Optional, Union, Any, Dict
from openai import AsyncOpenAI
from api.misc import parse_model_info
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
        super().__init__(base_url=base_url, api_key=api_key, **kwargs)
        self.default_model = kwargs.pop("model", None)
    
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
            responses.append(
                ChatCompletionResponseChoice(
                    index=i.index,
                    message=ChatCompletionMessage(
                        role=i.message.role,
                        content=i.message.content,
                        reasoning_content=i.message.reasoning
                    ),
                    finish_reason=i.finish_reason
                )
            )

        return ChatCompletionResponse(
            id='deepseek-response',
            model=response.model,
            choices=responses,
            usage=ChatCompletionResponseUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
        )
    
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
        
        response = await self.create_chat_completion_response(request,**kwargs)
        return response.choices[0]

    async def judge_answer_correctness(self,model_name,question, answer, llm_answer):
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

        judge_response = await self.create_chat_from_message(judge_prompt,model_name)
        if '\\boxed{correct}' in judge_response.message.content:
            return True
        else:
            return False
# TODO

# def parse_client(args:Optional[Dict[str,Any]] = None)-> List[Client]:
#     """
#     return : list of client, regardless of whether it is deployed one or a third-party api.
#     """
#     # client have the same base url should be regarded as the same client
#     model_args,data_args,generating_args,distill_args = get_infer_args(args)
#     clients = []
#     model_infos = parse_model_info(model_args)
#     for model_info in model_infos:
#         if model_info["template"] != "None":
#             # need to deploy locally
#         else:
#             pass
    
    
    
    