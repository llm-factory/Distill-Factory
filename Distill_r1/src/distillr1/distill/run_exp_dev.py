import os
import sys
import json
from typing import Dict, Any, List, Optional
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args,get_origin_infer_args
from ..data.loader import get_dataset,get_qa_pairs
from ..extras import logging
from ..api.client import Client,parse_client
from ..api.router import ModelRouter
from ..api.app import run_api
from .judge import SYSTEM_JUDGE_PROMPT
import asyncio
import debugpy
from .distiller import Distiller
logger = logging.get_logger(__name__)
async def run_exp(args: Optional[Dict[str, Any]] = None,max_try=3) -> None:
    """
    Run data loading experiment with given arguments.
    
    Args:
        args (Optional[Dict[str, Any]]): Optional dictionary of arguments to override defaults
    """
    router = run_api()
    model_infos = router.get_model_infos()    
    clients = parse_client(model_infos)
    model_args, data_args, finetuning_args,generating_args,distill_args = get_origin_infer_args(args)
    chat_client = clients["chat"]
    if distill_args.enable_reward_model:
        reward_client = clients["reward"]
    questions,answers = get_qa_pairs(model_args, data_args)
    distiller = Distiller(distill_args,clients,questions,answers)
    await distiller.distill()
    
    
    
    # if "train_dataset" in dataset_module:
    #     train_dataset_reasoner = []
    #     train_dataset = dataset_module["train_dataset"]
    #     logger.info(f"Source Dataset size: {len(train_dataset)}")
    #     logger.info(f"Source Dataset columns: {train_dataset.column_names}")
    #     if len(train_dataset) > 0:
    #         logger.info("First data example:")
    #         logger.info(train_dataset[0])
    #     for question, answer in zip(train_dataset["_prompt"], train_dataset["_response"]):
    #         question = question[0]['content']
    #         answer = answer[0]['content']
    #         llm_response = await chat_client.create_chat_from_message(distill_args.meta_prompt + question,model_args.model_name_or_path) # TODO-> 
    #         llm_answer = llm_response.message.content
    #         llm_reason = llm_response.message.reasoning_content
    #         for i in range(max_try):
    #             judge_result = await reward_client.judge_answer_correctness(question, answer, llm_answer)
    #             logger.info_rank0(f"judge_result:{judge_result}\tquestion:{question}\t answer:{answer}\t")
    #             if judge_result:
    #                 train_dataset_reasoner.append({
    #                     "instruction": question,
    #                     "input": "",
    #                     "output": f"<think>\n{llm_reason}\n</think>\n\n{llm_answer}"
    #                 })
    #                 with open("test.json", "w",encoding='utf-8') as f:
    #                     json.dump(train_dataset_reasoner,f,ensure_ascii=False,indent=2)      
    #                 break
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        task = loop.create_task(run_exp())
    else:
        asyncio.run(run_exp())
