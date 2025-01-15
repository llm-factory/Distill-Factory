from strategy.method import TextRetriever,BaseTextRetriever,BaseQAVerifier,Generator,Verifier
from strategy.strategy import Strategy
from model.config import Config
from common.message import *
from tools.tool import *
from tools.filter import *
from typing import List, Tuple, Dict, Any
import asyncio
import logging
import re
from tqdm import tqdm
import random

logger = logging.getLogger('logger')

DEFAULT_TITLE_PROMPT = """你是一个优秀的文本阅读与信息提炼助手。请根据下面提供的文本内容，提取出多个精准且具有针对性的小标题。这些小标题应满足以下要求：
1. 每个小标题必须包含确切的时间（若文本中有提及），地点，人物名称，事件名称或组织名称等具体信息。
2. 小标题必须清晰明了，不含模糊表达或歧义性描述。
3. 小标题应从整体上覆盖文本中的核心主题及重要信息点，确保不遗漏关键事件。
4. 每个小标题独占一行，不要有其他信息。
5. 直接输出小标题，不要有其他信息。"""

DEFAULT_QUESTION_REQ_PROMPT = """要求如下：
1. 问题必须包括完整的信息以避免模糊，例如：具体的人物、名称、事件、时间等。
2. 问题应当客观、具体，避免模糊不清。
3. 问题需基于客观事实，不得包含主观感受、预测或想象。问题应当能在文本中找到答案
4. 确保问题内容不重复，包含不同类型的问题并且覆盖文本的不同部分或不同维度
""" 

DEFAULT_QUESTION_FORMAT_CONTROL = "每个问题以'问题'加数字加'::'开头，且问题内容不能重复。"

DEFAULT_ANSWER_PROMPT = """你是一位专业的AI助手。请根据以下文本内容回答问题。对于无法回答的问题，请回复'无法回答'。"""

class TwoStageQAGenerator(Generator):
    def __init__(self, api, config,title_prompt: str = DEFAULT_TITLE_PROMPT,
                 question_prompt: str = DEFAULT_QUESTION_REQ_PROMPT,
                 answer_prompt: str = DEFAULT_ANSWER_PROMPT):
        self.api = api
        self.title_prompt = title_prompt
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.config = config
        self.split = self.config.quantity_level >= 4
        self.num_questions_per_title = self.config.quantity_level

    async def generate(self, text: str, config: GenerationConfig) -> Tuple[List[str], List[str]]:
        personas = []
        if config.diversity_mode == 'persona':
            personas = await self.getPersona(text)
        
        questions = await self._generate_questions(text, config,personas)
        
        answers = await self._generate_answers(text, questions, config)
        answers, idxs_to_remove = answers_filter(answers)
        questions = [q for idx, q in enumerate(questions) if idx not in idxs_to_remove]
        return questions, answers

    async def _generate_questions(self, text: str, config: GenerationConfig,personas: List[str]) -> List[str]:
        titles = await self._generate_titles(text,config)
        all_questions = []
        
        for i in range(0, len(titles), config.concurrent_requests):
            batch_titles = titles[i:i + config.concurrent_requests]
            prompts = []
            logger.debug(f"{'-'*20}Titles{'-'*20}")
            logger.debug(batch_titles)
            for title in batch_titles:
                prompt_template = (config.question_prompt or DEFAULT_QUESTION_REQ_PROMPT) + DEFAULT_QUESTION_FORMAT_CONTROL
                formatted_prompt = f"""
你对{config.main_theme}相关内容十分感兴趣。请您根据以下文本内容，围绕'{config.main_theme}'提出{self.num_questions_per_title}个清晰、客观的问题，"
这些问题必须紧密围绕'{title}'，且可以在文本中找到明确的答案。\n"""+prompt_template.format(
                )
                if len(personas) > 0:
                    formatted_prompt = f"你是{random.choice(personas)}" + formatted_prompt
                prompt = buildMessages(
                    SystemMessage(formatted_prompt),
                    UserMessage(f"文本: {text}")
                )
                prompts.append(prompt)

            logger.debug(f"{'-'*20}Prompts{'-'*20}")
            logger.debug(prompts)
            responses = await self.api.async_chat(prompts,temperature=self.config.temperature)
            responses = parse_responses(responses,"问题[\d ]+::")
            all_questions.extend(responses)
        
        return all_questions
    async def getPersona(self, text):
        prompt = buildMessages(
            SystemMessage(
                f"你是一个擅长生成人物描述的助手。"
                "以下是一个文本，你需要根据文本内容生成10个可能对该文本相关内容感兴趣的大致人物描述。\n"
                "生成的人物应具有普遍性、对文本大部分内容感兴趣。每个人物应有独特的兴趣或特点，不能包含人名，每个人物一行。\n"
            ),
            UserMessage(
                f"{text}"
            )
        )
        personas = await self.api.async_chat([prompt],temperature=self.config.temperature)
        personas = clean_and_split_reply_list(personas)
        return personas
    async def _generate_answers(self, text: str, questions: List[str], 
                              config: GenerationConfig) -> List[str]:
        answers = []
        
        for i in range(0, len(questions), config.concurrent_requests):
            batch_questions = questions[i:i + config.concurrent_requests]
            prompts = []
            
            for question in batch_questions:
                prompt_template = self.answer_prompt 
                # print(prompt_template)
                formatted_prompt = prompt_template.format()
                prompt = buildMessages(
                    SystemMessage(formatted_prompt),
                    UserMessage(f"文本: {text}\n问题: {question}" + config.answer_prompt if config.answer_prompt else f"文本: {text}\n问题: {question}")
                )
                prompts.append(prompt)
            
            genAnswers = await self.api.async_chat(prompts,temperature=self.config.temperature)
            
            for q,a in zip(batch_questions,genAnswers):
                logger.debug(f"{'-'*20}QA pair{'-'*20}")
                logger.debug(f"{'-'*15}Question{'-'*15}")
                logger.debug(f"{q}")
                logger.debug(f"{'-'*15}Answer{'-'*15}")
                logger.debug(f"{a}")
            answers.extend(genAnswers)
        
        return answers

    async def _generate_titles(self, text: str,config) -> List[str]:
        prompt = buildMessages(
            SystemMessage(self.title_prompt),
            UserMessage(text)
        )
        responses = await self.api.async_chat([prompt],temperature=self.config.temperature)
        titles = clean_and_split_title_list(responses)
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
        return splitTitles
class SimpleQAVerifier(Verifier):
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

class genQA(Strategy):
    def _create_text_retriever(self) -> TextRetriever:
        return BaseTextRetriever(self.api)
    
    def _create_qa_generator(self) -> Generator:
        return TwoStageQAGenerator(self.api,self.config)
    
    def _create_qa_verifier(self) -> Verifier:
        return SimpleQAVerifier(self.api)
    
    async def process_single_data(self, text: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        logger.debug('='*30 + 'Processing Text' + '='*30)
        logger.debug(text[:200])
        
        questions, answers = await self.qa_generator.generate(text, config)
        questions = questions_filter(questions)
        
        logger.debug(f"{'='*30}Questions{'='*30}")
        logger.debug(questions)
        logger.debug(f"{'='*30}Answers{'='*30}")
        logger.debug(answers)
        
        save_QA_dataset(questions, answers, config.save_dir, config.save_file_name,config.max_nums)

        return questions, answers

    async def process_single_file(self, file_path: str, config: Config) -> Tuple[List[str], List[str]]:
        texts = self.text_retriever.get_text(Path(file_path),config)
        
        all_questions = []
        all_answers = []
        tasks = []
        
        
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
        logger.debug(file_paths)
        all_questions = []
        all_answers = []        
        tasks = [self.process_single_file(file_path, config) for file_path in file_paths]
        concurrent_limit = 1 if config.is_structure_data else config.concurrent_api_requests_num
        for i in range(0, len(tasks), concurrent_limit):
            batch_tasks = tasks[i:i + concurrent_limit]
            results = await asyncio.gather(*batch_tasks)
            
            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)
        
        return all_questions, all_answers