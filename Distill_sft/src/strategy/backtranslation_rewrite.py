from strategy.strategy import Strategy
from strategy.method import *
from model.config import Config
from common.message import *
from tools.tool import *
from tools.filter import *
from typing import List, Tuple, Dict, Any
import asyncio
import logging
import jieba
from tqdm import tqdm
from common.prompts import *
logger = logging.getLogger('logger')

DEFAULT_QUESTION_PROMPT = """请基于以下事实，生成{num_questions_per_title}个清晰且能够依据该事实清晰正确回答的问题，问题需要覆盖事实的不同部分或不同维度。
事实:{extraction}
问题需包含充分的信息，如关键细节以及关键信息（如具体的名称、时间、地点、事件等），以避免提问模糊或不清晰。禁止使用模糊的指代词(如"这个","那个","它",'这次','这天'等)。
"""
class BacktransQAGenerator(Generator):
    def __init__(self, api,config,
                 title_prompt: str = DEFAULT_TITLE_EXTRACTION_PROMPT,
                 extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
                 question_prompt: str = DEFAULT_QUESTION_PROMPT,
                 answer_prompt: str = DEFAULT_ANSWER_PROMPT):
        self.api = api
        self.config = config
        self.title_prompt = title_prompt
        self.extraction_prompt = DEFAULT_EXTRACTION_PROMPT
        self.question_prompt = config.question_prompt
        self.answer_prompt = config.answer_prompt
        self.meaningless_symbols = [' ', '，', '。', '、', '：', '；', '"', '"', ''', ''', '(', ')', '（', '）', '《', '》', '【', '】', '!', '！', '?', '？', '——', '……']
        self.split = config.quantity_level>=4
        self.num_questions_per_title = config.quantity_level

    async def generate(self, text: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        titles = await self._generate_titles(text,config)
        logger.debug(f"{'-'*20}Titles{'-'*20}")
        logger.debug(titles)
        questions, factlist, extraction2questions = await self._get_factlist(
            config,text, titles
        )
        questions = questions_filter(questions)
        answers = await self._generate_answers(text, questions)
        answers, idxs_to_remove = answers_filter(answers)
        questions = [q for idx, q in enumerate(questions) if idx not in idxs_to_remove]
        
        rewritten_answers = await self._rewrite_answers(
            questions, answers, text, config.concurrent_api_requests_num
        )
        
        return questions, rewritten_answers

    async def _generate_titles(self, text: str,config) -> List[str]:
        prompt = buildMessages(
            SystemMessage(self.title_prompt),
            UserMessage(text)
        )
        responses = await self.api.async_chat([prompt])
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

    async def _get_factlist(self, config,text: str, titles: List[str]) -> Tuple[List[str], List[str], List[Dict]]:
        extraction2questions = []
        factlist = []
        questions = []
        
        for idx in range(0, len(titles), config.concurrent_api_requests_num):
            batch_titles = titles[idx:idx + config.concurrent_api_requests_num]
            batch_prompts = []
            
            for title in batch_titles:
                prompt = buildMessages(
                    SystemMessage(self.extraction_prompt.format(main_theme=config.main_theme)+DEFAULT_EXTRACTION_FORMAT),
                    UserMessage(f"文本：{text}\n标题：{title}"
                    )
                )
                batch_prompts.append(prompt)
                
            batch_extractions = await self.api.async_chat(batch_prompts,temperature=config.temperature)
            logger.debug(f"{'-'*20}Extractions{'-'*20}")
            logger.debug(batch_extractions)
            for title, extractions_text in zip(batch_titles, batch_extractions):
                titleset = jieba.cut_for_search(title)
                titleset = list(";".join(titleset).split(';'))
                titleset = [t for t in titleset if t not in self.meaningless_symbols]
                titleset = [t for t in titleset if len(t) >= 3 and len(t) <= 30]
                
                main_theme_set = jieba.cut_for_search(config.main_theme)
                main_theme_set = list(";".join(main_theme_set).split(';'))
                
                extractions = extract_json(extractions_text,"extraction")
                extractions = [e for es in extractions for e in es]
                if not extractions:
                    extractions = [extractions_text]
                extractions = [e for e in extractions if any(theme in e for theme in (main_theme_set + titleset))]
                for ext_idx in range(0, len(extractions), config.concurrent_api_requests_num):
                    batch_extractions = extractions[ext_idx:ext_idx + config.concurrent_api_requests_num]
                    question_prompts = []
                    
                    for extraction in batch_extractions:
                        question_prompt = DEFAULT_QUESTION_PROMPT + self.question_prompt if self.question_prompt else DEFAULT_QUESTION_PROMPT
                        
                        prompt = buildMessages(
                            UserMessage(question_prompt.format(extraction=extraction,num_questions_per_title=self.num_questions_per_title) + DEFAULT_QUESTION_FORMAT)
                        )
                        question_prompts.append(prompt)
                        
                    batch_gen_questions = await self.api.async_chat(question_prompts,temperature=config.temperature)
                    
                    for extraction, gen_questions in zip(batch_extractions, batch_gen_questions):
                        gen_questions = extract_json(gen_questions,"question")
                        valid_questions = await self._validate_questions(*gen_questions, config.concurrent_api_requests_num)
                        logger.debug(f"{'-'*20}Generated Questions{'-'*20}")
                        for q in gen_questions:
                            logger.debug(q)
                        if len(valid_questions) > 0:
                            extraction2questions.append({
                                "extraction": extraction,
                                "questions": valid_questions
                            })
                            questions.extend(valid_questions)
        return questions, factlist, extraction2questions

    async def _validate_questions(self, questions: List[str], concurrent_requests: int) -> List[str]:
        valid_questions = []
        for q_idx in range(0, len(questions), concurrent_requests):
            batch_questions = questions[q_idx:q_idx + concurrent_requests]
            validation_prompts = []
            
            for q in batch_questions:
                prompt = buildMessages(
                    UserMessage(DEFAULT_JUDGE_PROMPT.format(question=q))
                )
                validation_prompts.append(prompt)
            
            validation_results = await self.api.async_chat(validation_prompts, temperature=0.7)
            
            for q, result in zip(batch_questions, validation_results):
                if "【有效】" in result.split('\n')[-1]:
                    valid_questions.append(q)
                    
        return valid_questions

    async def _generate_answers(self,text: str, questions: List[str]) -> List[str]:
        answers = []
        for idx in range(0, len(questions), self.config.concurrent_api_requests_num):
            batch_questions = questions[idx:idx + self.config.concurrent_api_requests_num]
            prompts = []
            answer_prompt = DEFAULT_ANSWER_PROMPT
            for question in batch_questions:
                
                prompt = buildMessages(
                    SystemMessage(DEFAULT_ANSWER_PROMPT + self.answer_prompt if self.answer_prompt else DEFAULT_ANSWER_PROMPT),
                    UserMessage(DEFAULT_QUESTION_ANSWERING_TEMPLATE.format(text=text,question=question))
                )
                prompts.append(prompt)
                
            batch_answers = await self.api.async_chat(prompts,temperature=self.config.temperature)
            
            for q, a in zip(batch_questions, batch_answers):
                logger.debug(f"{'-'*20}QA pairs{'-'*20}")
                logger.debug(f"{'-'*15}Question{'-'*15}")
                logger.debug(f"{q}")
                logger.debug(f"{'-'*15}Answer{'-'*15}")
                logger.debug(f"{a}")
                
            answers.extend(batch_answers)
            
        return answers

    async def _rewrite_answers(self, questions: List[str], answers: List[str], 
                             text: str, concurrent_requests: int) -> List[str]:
        new_answers = []
        for idx in range(0, len(questions), concurrent_requests):
            batch_questions = questions[idx:idx + concurrent_requests]
            prompts = []
            
            for q in batch_questions:
                prompt = buildMessages(
                    SystemMessage(""),
                    UserMessage(f"""文本：{text}
问题：{q}
请根据所给文本高质量地回答上述问题，回答应正确、通顺、清晰，据有深度。不应出现"根据文本","文本中"等字眼。""" + self.answer_prompt)
                )
                prompts.append(prompt)
            batch_answers = await self.api.async_chat(prompts,temperature=self.config.temperature)
            for q, a in zip(batch_questions, batch_answers):
                logger.debug(f"{'-'*20}rewritten QA pairs{'-'*20}")
                logger.debug(f"{'-'*15}Question{'-'*15}")
                logger.debug(f"{q}")
                logger.debug(f"{'-'*15}Answer{'-'*15}")
                logger.debug(f"{a}")
            new_answers.extend(batch_answers)
        return new_answers

class backtranslation_rewrite(Strategy):
    def _create_text_retriever(self) -> TextRetriever:
        return BaseTextRetriever(self.api,self.config)
    
    def _create_qa_generator(self) -> Generator:
        return BacktransQAGenerator(self.api,self.config)
    
    def _create_qa_verifier(self) -> Verifier:
        return BaseQAVerifier(self.api)
    
    async def process_single_data(self, text: str, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        logger.debug('='*30 + 'Processing Text' + '='*30)
        logger.debug(text[:200])
        
        questions, answers = await self.qa_generator.generate(text, config)
        save_QA_dataset(questions, answers, config.save_dir, config.save_file_name,config.max_nums)
        
        return questions, answers

    async def process_single_file(self, file_path: str, config: Config) -> Tuple[List[str], List[str]]:
        texts = self.text_retriever.get_text(Path(file_path), config)
        
        all_questions = []
        all_answers = []
        tasks = []
        
        for text in texts:
            tasks.append(self.process_single_data(text, config))
            
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
        
        return all_questions, all_answers

