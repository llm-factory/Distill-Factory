import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union, List


@dataclass
class ReasoningCurationArguments:
    r"""
    """
    # TODO
    min_reasoning_length: Optional[int] = field(
        default=None,
        metadata={"help": "The minimum length of reasoning for synthetic data generation."},
    )
    max_reasoning_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of reasoning for synthetic data generation."},
    )
    reasoning_length_preference: Optional[Literal["shortest", "longest"]] = field(
        default=None,
        metadata={"help": "The preference for the length of reasoning content."},
    )
    

@dataclass
class ResponseCurationArguments:
    r"""
    """
    # TODO
    min_response_length: Optional[int] = field(
        default=None,
        metadata={"help": "The minimum length of response for synthetic data generation."},
    )
    max_response_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum length of response for synthetic data generation."},
    )
    response_length_preference: Optional[Literal["shortest", "longest"]] = field(
        default=None,
        metadata={"help": "The preference for the length of response content."},
    )
    exact_match: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use exact match."},
    )
    rouge: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use rouge."},
    )

@dataclass
class CurationArguments(ReasoningCurationArguments,ResponseCurationArguments):
    """
    Arguments for the synthetic data curation.
    """
    curation: bool = field(
        default=True,
        metadata={"help": "Whether or not to curate synthetic data."},
    )
    llm_as_judge: bool = field(
        default=False,
        metadata={"help": "Whether or not to use LLM as judge."},
    )
    length_filter: bool = field(
        default=False,
        metadata={"help": "Whether or not to use length filter."},
    )
    
    
    def __post_init__(self):
        self.length_filter = True if any([
            self.min_reasoning_length is not None,self.max_reasoning_length is not None,
            self.min_response_length is not None,self.max_response_length is not None,
        ]) else False