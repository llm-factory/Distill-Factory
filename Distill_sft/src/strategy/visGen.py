import base64

from strategy.method import TextRetriever, BaseTextRetriever, BaseQAVerifier, Generator, Verifier
from strategy.strategy import Strategy
from model.config import Config
from common.message import *
from tools.doc_render.pdf import pdf_to_images
from tools.tool import *
from tools.filter import *
from typing import List, Tuple, Dict, Any
import asyncio
import logging
from pathlib import Path
import json
from tqdm import tqdm
from common.prompts import *

logger = logging.getLogger('logger')

DEFAULT_QA_REQ_PROMPT = """请根据图中所示文档提出{num_qa}条问题并回答对应的答案。要求如下：
1. 问题必须包括完整的信息以避免模糊，例如：具体的人物、名称、事件、时间等。
2. 问题应当客观、具体，避免模糊不清。
3. 问题需基于客观事实，不得包含主观感受、预测或想象。问题应当能在文本中找到答案。
4. 答案必须准确、完整，直接回答问题，不要有无关内容。
"""

DEFAULT_QA_FORMAT = """
需要使用json格式输出,格式示例如下:
[
    {
        "question": "问题1",
        "answer": "问题1的对应答案"
    },
    {
        "question": "问题2",
        "answer": "问题2的对应答案"
    },
]
"""


class VisGenQAGenerator(Generator):
    def __init__(self, api, config):
        self.api = api
        self.title_prompt = DEFAULT_TITLE_EXTRACTION_PROMPT
        self.qa_prompt = DEFAULT_QA_REQ_PROMPT
        self.config = config
        self.split = self.config.quantity_level >= 4
        self.num_qa = self.config.quantity_level
        self.question_prompt = config.question_prompt if config.question_prompt else ""
        self.answer_prompt = config.answer_prompt if config.answer_prompt else ""
        self.text_retriever = None
        self.image_renderer = None

    async def generate(self, image: bytes, config: GenerationConfig) -> Tuple[List[str], List[str]]:
        all_questions = []
        all_answers = []

        questions, answers = await self._generate_qa_pairs(image, config)
        all_questions.extend(questions)
        all_answers.extend(answers)

        return all_questions, all_answers

    async def _generate_qa_pairs(self, image: bytes, config: GenerationConfig) -> List[Tuple[List[str], List[str]]]:
        formatted_prompt = self.qa_prompt.format(num_qa=self.num_qa) \
                           + self.question_prompt + self.answer_prompt + DEFAULT_QA_FORMAT

        encode_image = base64.b64encode(image).decode('utf-8')

        prompt = buildMessages(
            SystemMessage(formatted_prompt),
            UserMessage([
                {"type": "text", "text": "待抽取的文档内容是："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image}"}}
            ])
        )

        responses = await self.api.async_chat([prompt], temperature=self.config.temperature)
        logger.info("prompt")
        logger.info(prompt)
        questions, answers = extract_json(responses, "question", "answer")
        logger.info(f"{'-' * 20}Questions{'-' * 20}")
        logger.info(questions)
        logger.info(f"{'-' * 20}Answers{'-' * 20}")
        logger.info(answers)
        return questions, answers


class VisGenQAVerifier(Verifier):
    def __init__(self, api):
        self.api = api
        self.verifier = BaseQAVerifier(api)

    async def verify(self, text: str, questions: List[str], answers: List[str],
                     config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        return await self.verifier.verify(
            text,
            config.get('main_theme', ''),
            questions,
            answers,
            config.get('concurrent_api_requests_num', 1)
        )


class VisGenQA(Strategy):
    def _create_text_retriever(self) -> TextRetriever:
        return BaseTextRetriever(self.api, self.config)

    def _create_qa_generator(self) -> Generator:
        return VisGenQAGenerator(self.api, self.config)

    def _create_qa_verifier(self) -> Verifier:
        return VisGenQAVerifier(self.api)

    async def process_single_data(self, image: bytes, config: Config) -> Tuple[List[str], List[str]]:
        """对一张图生成 QA Pairs"""
        logger.debug('=' * 30 + 'Processing Images' + '=' * 30)

        questions, answers = await self.qa_generator.generate(image, config)
        questions = questions_filter(questions)

        logger.debug(f"{'=' * 30}Questions{'=' * 30}")
        logger.debug(questions)
        logger.debug(f"{'=' * 30}Answers{'=' * 30}")
        logger.debug(answers)

        save_QA_dataset(questions, answers, config.save_dir, config.save_file_name, config.max_nums)

        return questions, answers

    async def process_single_file(self, file_path: Path, config: Config) -> Tuple[List[str], List[str]]:
        """对一个文件生成 QA Pairs"""
        with open(file_path, 'rb') as file:
            file_data = file.read()
        images = pdf_to_images(file_data)
        all_questions = []
        all_answers = []

        for i in range(0, len(images), config.concurrent_api_requests_num):
            batch_images = images[i:i + config.concurrent_api_requests_num]
            # TODO
            tasks = [self.process_single_data(image, config) for image in batch_images]
            results = await asyncio.gather(*tasks)

            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)

        return all_questions, all_answers

    async def run(self, config: Config) -> Tuple[List[str], List[str]]:
        init_QA_dataset(config.save_dir, config.save_file_name)

        file_paths = getFilePaths(config)
        logger.debug(f"{'=' * 30}File Paths{'=' * 30}")
        logger.info(file_paths)

        all_questions = []
        all_answers = []

        concurrent_limit = config.concurrent_api_requests_num
        tasks = [self.process_single_file(file_path, config) for file_path in file_paths]

        for i in range(0, len(tasks), concurrent_limit):
            batch_tasks = tasks[i:i + concurrent_limit]
            results = await asyncio.gather(*batch_tasks)

            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)

        return all_questions, all_answers
