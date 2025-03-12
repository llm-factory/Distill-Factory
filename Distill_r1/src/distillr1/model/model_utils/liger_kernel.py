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

import inspect
from typing import TYPE_CHECKING

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    from liger_kernel.transformers import LigerCrossEntropyLoss, LigerRMSNorm, LigerSwiGLUMLP
    from liger_kernel.transformers.model.qwen2_vl import lce_forward as qwen2_vl_lce_forward
    from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl

    def get_dtype(self: "modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel"):
        return self.dtype

    modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.get_dtype = get_dtype

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb

    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        modeling_qwen2_5_vl.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_vl_lce_forward

    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP
