# Copyright 2024 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
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

import asyncio
import os
from threading import Thread
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from ..extras.misc import torch_gc
from ..hparams import get_infer_args,get_origin_infer_args
from .base_engine import BaseEngine,Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""
    General class for chat models. Backed by huggingface or vllm engines.

    Supports both sync and async methods.
    Sync methods: chat(), stream_chat() and get_scores().
    Async methods: achat(), astream_chat() and aget_scores().
    """

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args,_ = get_origin_infer_args(args)
        self.engine_type = model_args.infer_backend
        if model_args.infer_backend == "huggingface":
            from .hf_engine import HuggingfaceEngine
            self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, generating_args)
        elif model_args.infer_backend == "vllm":
            from .vllm_engine import VllmEngine
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        r"""
        Gets a list of responses of the chat model.
        """
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, videos, audios, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[Sequence["ImageInput"]] = None,
        videos: Optional[Sequence["VideoInput"]] = None,
        audios: Optional[Sequence["AudioInput"]] = None,
        **input_kwargs,
    ) -> List["Response"]:
        r"""
        Asynchronously gets a list of responses of the chat model.
        """
        return await self.engine.chat(messages, system, tools, images, videos, audios, **input_kwargs)