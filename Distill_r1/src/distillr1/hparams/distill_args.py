from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

@dataclass
class CurationArguments:
    """
    Arguments for the synthetic data curation.
    TODO
    """
    pass

@dataclass
class DistillArguments(CurationArguments):
    """
    Arguments for the synthetic data generation.
    """
    method: str = field(
        default="basic",
        metadata={"help": "Method to generate synthetic data.(Reasoning? ... )"},
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "The output directory where the synthetic dataset will be written."},
    )
    output_path: Optional[str] = field(
        default="./synthetic_dataset.json",
        metadata={"help": "The output path where the synthetic dataset will be written."},
    )
    meta_prompt: Optional[str] = field(
        default="",
        metadata={"help": "The meta prompt for the synthetic dataset."},
    )