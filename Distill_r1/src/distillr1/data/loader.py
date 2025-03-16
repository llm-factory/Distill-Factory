# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union,Tuple,List

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk

from ..extras import logging
from ..extras.misc import check_version
from .converter import align_dataset
from .data_utils import merge_dataset, split_dataset
from .parser import get_dataset_list
from ..extras.constants import FILEEXT2TYPE


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr

logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments"=None,
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
        )

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args)


def _get_merged_dataset(
    dataset_names: Optional[Sequence[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    merge: bool = True,
) -> Optional[Union["Dataset", "IterableDataset", Dict[str, "Dataset"]]]:
    r"""
    Returns the merged datasets in the standard format.
    """
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        # ...
        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args)

    if merge:
        return merge_dataset(list(datasets.values()))
    else:
        return datasets

def get_dataset(
    model_args: "ModelArguments",
    data_args: "DataArguments"
) -> "DatasetModule":
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    dataset = _get_merged_dataset(data_args.dataset, model_args, data_args)

    dataset_dict = {}
    if dataset is not None:
        dataset_dict["train"] = dataset
    dataset_dict = DatasetDict(dataset_dict)

    dataset_module = {}
    if "train" in dataset_dict:
        dataset_module["train_dataset"] = dataset_dict["train"]

    if "validation" in dataset_dict:
        dataset_module["eval_dataset"] = dataset_dict["validation"]
    else:
        eval_dataset = {}
        for key in dataset_dict.keys():
            if key.startswith("validation_"):
                eval_dataset[key[len("validation_") :]] = dataset_dict[key]

        if len(eval_dataset):
            dataset_module["eval_dataset"] = eval_dataset

    return dataset_module

def get_qa_pairs(
    model_args: "ModelArguments",
    data_args: "DataArguments",
) -> Tuple[List[str],List[str]]:
    r"""
    Gets the question-answer pairs from the dataset.
    """
    dataset_module = get_dataset(model_args, data_args)
    if "train_dataset" in dataset_module:
        train_dataset = dataset_module["train_dataset"]
        logger.info(f"Source Dataset size: {len(train_dataset)}")
        logger.info(f"Source Dataset columns: {train_dataset.column_names}")
        if len(train_dataset) > 0:
            logger.info("First data example:")
            logger.info(train_dataset[0])
        return train_dataset["_prompt"], train_dataset["_response"]