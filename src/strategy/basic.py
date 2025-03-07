from strategy.method import TextRetriever, BaseTextRetriever, BaseQAVerifier, Generator, Verifier
from strategy.strategy import Strategy
from model.config import Config
from common.message import *
from tools.tool import *
from tools.filter import *
from typing import List, Tuple, Dict, Any
import asyncio
import logging
from pathlib import Path
import json
from tqdm import tqdm

logger = logging.getLogger('logger')

DEFAULT_TITLE_PROMPT = """你是一个优秀的文本阅读与信息提炼助手。请根据下面提供的文本内容，提取出多个精准且具有针对性的小标题。这些小标题应满足以下要求：
1. 每个小标题必须包含确切的时间（若文本中有提及），地点，人物名称，事件名称或组织名称等具体信息。
2. 小标题必须清晰明了，不含模糊表达或歧义性描述。
3. 小标题应从整体上覆盖文本中的核心主题及重要信息点，确保不遗漏关键事件。
4. 每个小标题独占一行，不要有其他信息。
5. 直接输出小标题，不要有其他信息。"""

DEFAULT_QA_REQ_PROMPT = """请根据所给文本指向{title}提出{num_qa}条问题并回答对应的答案。要求如下：
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

class BasicQAGenerator(Generator):
    def __init__(self, api, config):
        self.api = api
        self.title_prompt = DEFAULT_TITLE_PROMPT
        self.qa_prompt = DEFAULT_QA_REQ_PROMPT
        self.config = config
        self.split = self.config.quantity_level >= 4
        self.num_qa = self.config.quantity_level
        self.question_prompt = config.question_prompt if config.question_prompt else ""
        self.answer_prompt = config.answer_prompt if config.answer_prompt else ""
        self.text_retriever = BaseTextRetriever(self.api,self.config)

    async def generate(self, text: str, config: GenerationConfig) -> Tuple[List[str], List[str]]:
        titles = await self._generate_titles(text,config)
        logger.debug(f"{'-'*20}Generated Titles{'-'*20}")
        logger.debug(titles)
        
        all_questions = []
        all_answers = []
        
        for i in range(0, len(titles), config.concurrent_requests):
            batch_titles = titles[i:i + config.concurrent_requests]
            questions,answers = await self._generate_qa_pairs(text, batch_titles, config)
            all_questions.extend(questions)
            all_answers.extend(answers)
        
        return all_questions, all_answers

    async def _generate_titles(self, text: str,config) -> List[str]:
        prompt = buildMessages(
            SystemMessage(self.title_prompt),
            UserMessage(text)
        )
        responses = await self.api.async_chat([prompt],temperature=self.config.temperature)
        logger.info(f"titles of:{text[:50]}")
        logger.info(responses)
        titles = clean_and_split_title_list(responses)
        logger.info(f"{'-'*20}Titles after clean{'-'*20}")
        logger.info(titles)
        if self.split:
            return await self.splitTitles(titles,config.concurrent_api_requests_num)
        else:
            return titles
        
    async def splitTitles(self,titles,concurrent_api_requests_num=1):
        splitTitles = []
        for idx in tqdm(range(0,len(titles),concurrent_api_requests_num),desc='Splitting titles'):
            batch_titles = titles[idx:idx+concurrent_api_requests_num]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                        SystemMessage(
                        f"你是一个擅长划分标题的助手。以下是一个标题,该标题可能包含多个事实，不够简洁。对于包含多个事实的标题你需要将该标题划分为多个小标题,每个小标题要包含原标题中的核心信息和一部分有效信息，不能改变原意，每个小标题一行,不输出额外信息。对于已经足够简洁的标题，则输出原标题。"    
                        ),
                        UserMessage(
                            f"""标题: {batch_titles[i]}\n 只输出划分后的小标题或者原标题，不要有其他信息。"""
                        )
                )
                prompts.append(prompt)
            titles = await self.api.async_chat(prompts,temperature=self.config.temperature)
            titles = clean_and_split_title_list(titles)
            splitTitles.extend(titles)
        splitTitles = list(set(splitTitles))
        logger.info(f"{'-'*20}Split Titles{'-'*20}")
        logger.info(splitTitles)
        return splitTitles
    
    async def _generate_qa_pairs(self, text: str, titles: List[str], config: GenerationConfig) -> List[Tuple[List[str], List[str]]]:
        prompts = []
        for title in titles:
            formatted_prompt = self.qa_prompt.format(num_qa=self.num_qa,title=title) + self.question_prompt + self.answer_prompt +  DEFAULT_QA_FORMAT
            prompt = buildMessages(
                SystemMessage(formatted_prompt),
                UserMessage(f"标题: {title}\n文本: {text}")
            )
            prompts.append(prompt)

        responses = await self.api.async_chat(prompts, temperature=self.config.temperature)
        logger.info("prompts")
        logger.info(prompts)
        questions,answers = extract_json(responses,"question","answer")
        logger.info(f"{'-'*20}Questions{'-'*20}")
        logger.info(questions)
        logger.info(f"{'-'*20}Answers{'-'*20}")
        logger.info(answers)
        return questions,answers

class BasicQAVerifier(Verifier):
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

class BasicQA(Strategy):
    def _create_text_retriever(self) -> TextRetriever:
        return BaseTextRetriever(self.api, self.config)
    
    def _create_qa_generator(self) -> Generator:
        return BasicQAGenerator(self.api, self.config)
    
    def _create_qa_verifier(self) -> Verifier:
        return BasicQAVerifier(self.api)
    
    async def process_single_data(self, text: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        logger.debug('='*30 + 'Processing Text' + '='*30)
        logger.debug(text[:200])
        
        questions, answers = await self.qa_generator.generate(text, config)
        questions = questions_filter(questions)
        
        logger.debug(f"{'='*30}Questions{'='*30}")
        logger.debug(questions)
        logger.debug(f"{'='*30}Answers{'='*30}")
        logger.debug(answers)
        
        save_QA_dataset(questions, answers, config.save_dir, config.save_file_name, config.max_nums)
        
        return questions, answers

    async def process_single_file(self, file_path: str, config: Config) -> Tuple[List[str], List[str]]:
        texts = self.text_retriever.get_text(Path(file_path), config)        
        all_questions = []
        all_answers = []
        
        for i in range(0, len(texts), config.concurrent_api_requests_num):
            batch_texts = texts[i:i + config.concurrent_api_requests_num]
            tasks = [self.process_single_data(text, config) for text in batch_texts]
            results = await asyncio.gather(*tasks)
            
            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)
        
        return all_questions, all_answers

    async def run(self, config: Config) -> Tuple[List[str], List[str]]:
        init_QA_dataset(config.save_dir, config.save_file_name)
        
        file_paths = getFilePaths(config)
        logger.debug(f"{'=' * 30}File Paths{'='*30}")
        logger.info(file_paths)
        
        all_questions = []
        all_answers = []
        
        concurrent_limit = 1 if config.is_structure_data else config.concurrent_api_requests_num
        tasks = [self.process_single_file(file_path, config) for file_path in file_paths]
        
        for i in range(0, len(tasks), concurrent_limit):
            batch_tasks = tasks[i:i + concurrent_limit]
            results = await asyncio.gather(*batch_tasks)
            
            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)
        
        return all_questions, all_answers