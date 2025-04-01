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
from common.prompts import *
logger = logging.getLogger('logger')

class TwoStageQAGenerator(Generator):
    def __init__(self, api, config):
        self.api = api
        self.title_prompt = DEFAULT_TITLE_EXTRACTION_PROMPT
        self.question_prompt = config.question_prompt
        self.answer_prompt = config.answer_prompt if config.answer_prompt else DEFAULT_ANSWER_PROMPT
        self.config = config
        self.split = self.config.quantity_level >= 4
        self.num_questions_per_title = self.config.quantity_level
        self.text_retriever = BaseTextRetriever(self.api,self.config)
        self.qa_verifier = BaseQAVerifier(self.api)

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
                prompt_template = (config.question_prompt or DEFAULT_QUESTION_REQUIREMENT_PROMPT) + DEFAULT_QUESTION_FORMAT
                formatted_prompt = f"""
你对{config.main_theme}相关内容十分感兴趣。请您根据以下文本内容，围绕'{config.main_theme}'提出{self.num_questions_per_title}个清晰、客观的问题，"
这些问题必须紧密围绕'{title}'，且可以在文本中找到明确的答案。\n"""+prompt_template
                if len(personas) > 0:
                    formatted_prompt = f"你是{random.choice(personas)}" + formatted_prompt
                prompt = buildMessages(
                    SystemMessage(formatted_prompt),
                    UserMessage(f"文本: {text}")
                )
                prompts.append(prompt)

            responses = await self.api.async_chat(prompts,temperature=self.config.temperature)
            for title,response in zip(batch_titles,responses):
                logger.debug(f"{'-'*20}Title{'-'*20}")
                logger.debug(title)
                logger.debug(f"{'-'*20}Questions{'-'*20}")
                logger.debug(response)                
            questions = extract_json(responses,"question")
            all_questions.extend(*questions)
        
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
            if config.enable_rag:
                logger.info("enable rag")
                texts = await self.text_retriever.get_text_from_rag(batch_questions)
            else:
                texts = [text] * len(batch_questions)
            for question,text in zip(batch_questions,texts):
                prompt_template = self.answer_prompt 
                formatted_prompt = prompt_template.format()
                prompt = buildMessages(
                    SystemMessage(formatted_prompt+"根据上下文信息回答" if config.enable_rag else formatted_prompt),
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
                            DEFAULT_TITLE_SPLITTING_PROMPT
                        ),
                        UserMessage(
                            DEFAULT_TITLE_SPLITTING_TEMPLATE.format(title=batch_titles[i])
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
        return BaseTextRetriever(self.api,self.config)
    
    def _create_qa_generator(self) -> Generator:
        return TwoStageQAGenerator(self.api,self.config)
    
    def _create_qa_verifier(self) -> Verifier:
        return SimpleQAVerifier(self.api)
    
    async def process_single_data(self, text: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        logger.debug('='*30 + 'Processing Text' + '='*30)
        logger.debug(text[:200])
        
        questions, answers = await self.qa_generator.generate(text, config)
        questions = questions_filter(questions)
        if self.config.verify_qa:
            questions, answers = await self.qa_verifier.verify(text, questions, answers, config)
        
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
        logger.info(file_paths)
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
        
        logger.info("generation finished")
        return all_questions, all_answers