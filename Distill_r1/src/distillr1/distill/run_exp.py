import os
import sys
import json
from typing import Dict, Any, List, Optional
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args
from ..data.loader import get_dataset
from ..extras import logging
from ..api.client import Client
from .judge import SYSTEM_JUDGE_PROMPT
import asyncio
logger = logging.get_logger(__name__)


async def run_exp(args: Optional[Dict[str, Any]] = None,max_try=3) -> None:
    """
    Run data loading experiment with given arguments.
    
    Args:
        args (Optional[Dict[str, Any]]): Optional dictionary of arguments to override defaults
    """
    args = read_args(args)
    model_args,data_args,generating_args,distill_args = get_infer_args(args)
    client = Client(model_args.base_url, model_args.api_key)
    dataset_module = get_dataset(model_args, data_args)
    print(dataset_module)
    if "train_dataset" in dataset_module:
        train_dataset_reasoner = []
        train_dataset = dataset_module["train_dataset"]
        logger.info(f"Source Dataset size: {len(train_dataset)}")
        logger.info(f"Source Dataset columns: {train_dataset.column_names}")
        if len(train_dataset) > 0:
            logger.info("First data example:")
            logger.info(train_dataset[0])
        for question, answer in zip(train_dataset["_prompt"], train_dataset["_response"]):
            question = question[0]['content']
            answer = answer[0]['content']
            llm_response = await client.create_chat_from_message(distill_args.meta_prompt + question,model_args.model_name_or_path,**(generating_args.to_dict()))
            llm_answer = llm_response.message.content
            llm_reason = llm_response.message.reasoning_content
            for i in range(max_try):
                judge_result = await client.judge_answer_correctness(model_args.model_name_or_path,question, answer, llm_answer)
                logger.info_rank0(f"judge_result:{judge_result}\tquestion:{question}\t answer:{answer}\t")
                if judge_result:
                    train_dataset_reasoner.append({
                        "instruction": question,
                        "input": "",
                        "output": f"<think>\n{llm_reason}\n</think>\n\n{llm_answer}"
                    })
                    with open(distill_args.output_path, "w",encoding='utf-8') as f:
                        json.dump(train_dataset_reasoner,f,ensure_ascii=False,indent=2)      
                    break

        # TODO: save the dataset
