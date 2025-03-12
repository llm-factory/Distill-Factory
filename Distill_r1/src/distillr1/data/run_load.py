import os
import sys
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader
from ..hparams import ModelArguments, DataArguments, get_infer_args, read_args
from ..data.loader import get_dataset
from ..extras import logging
import tqdm
logger = logging.get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None) -> None:
    """
    Run data loading experiment with given arguments.
    
    Args:
        args (Optional[Dict[str, Any]]): Optional dictionary of arguments to override defaults
    """
    args = read_args(args)
    print(args)
    model_args, data_args,finetuning_args,generating_args,distill_args= get_infer_args(args)

    dataset_module = get_dataset(model_args, data_args)

    if "train_dataset" in dataset_module:
        train_dataset = dataset_module["train_dataset"]
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Training dataset columns: {train_dataset.column_names}")
        
        # Print a sample
        if len(train_dataset) > 0:
            logger.info("First training example:")
            logger.info(train_dataset[0])
                    
        for question,answer in zip(train_dataset["_prompt"],train_dataset["_response"]):
            print("Q\n")
            print(question)
            print("A")
            print(answer)

    if "eval_dataset" in dataset_module:
        eval_dataset = dataset_module["eval_dataset"]
        if isinstance(eval_dataset, dict):
            for key, dataset in eval_dataset.items():
                logger.info(f"Evaluation dataset '{key}' size: {len(dataset)}")
                logger.info(f"Evaluation dataset '{key}' columns: {dataset.column_names}")
        else:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            logger.info(f"Evaluation dataset columns: {eval_dataset.column_names}")
        print(eval_dataset)
            

if __name__ == "__main__":
    run_exp()