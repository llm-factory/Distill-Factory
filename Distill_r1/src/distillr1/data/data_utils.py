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

from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, TypedDict, Union
from datasets import DatasetDict,concatenate_datasets
from ..extras import logging

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = Sequence[Union[str, Set[str], Dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset"]]


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]], data_args: "DataArguments"=None, seed: int=None
) -> Union["Dataset", "IterableDataset"]:
    r"""
    Merges multiple datasets to a unified dataset.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        # concat by default
        return concatenate_datasets(all_datasets)
    
def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", seed: int
) -> "DatasetDict":
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
    dataset = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})