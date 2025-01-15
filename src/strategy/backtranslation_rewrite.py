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

logger = logging.getLogger('logger')

DEFAULT_TITLE_PROMPT = """你是一个优秀的文本阅读助手，请根据所给文本提取多个具有针对性的小标题。小标题必须包含具体的准确信息，例如准确的时间、地点、人物、名称、事件等。注意，你所提取的小标题不能指向模糊，不能有歧义。每个小标题一行，不要有重复."""

DEFAULT_EXTRACTION_PROMPT = """作为一个AI阅读理解助手，你将在下列给定文本中，提取与给定标题相关的关键信息
你必须严格遵循以下规则：
1.每条关键信息必须与标题相关，充分包含标题相关的信息。
2.每条关键信息必须包括{main_theme}相关字样。
3.每条关键信息不能重复。
每条关键信息以'关键'加数字加'::'开头。"""

DEFAULT_QUESTION_PROMPT = """请基于以下事实，生成{num_questions_per_title}个清晰且能够依据该事实清晰正确回答的问题，问题需要覆盖事实的不同部分或不同维度。
事实:{extraction}
问题需包含充分的信息，如关键细节以及关键信息（如具体的名称、时间、地点、事件等），以避免提问模糊或不清晰。禁止使用模糊的指代词(如"这个","那个","它",'这次','这天'等)。每个问题以'问题'加数字加'::'开头。"""

DEFAULT_ANSWER_PROMPT = """你是一个AI对话助手，你擅长从文本中提取信息并且高质量地回答人们的问题。
请根据文本回答问题。
回答中不要出现'根据文本'，'文本提到','文本中'等字样。"""

class BacktransQAGenerator(Generator):
    def __init__(self, api,config,
                 title_prompt: str = DEFAULT_TITLE_PROMPT,
                 extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
                 question_prompt: str = DEFAULT_QUESTION_PROMPT,
                 answer_prompt: str = DEFAULT_ANSWER_PROMPT):
        self.api = api
        self.config = config
        self.title_prompt = title_prompt
        self.extraction_prompt = extraction_prompt
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
        logger.info("after filt",questions)
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

    async def _get_factlist(self, config,text: str, titles: List[str]) -> Tuple[List[str], List[str], List[Dict]]:
        extraction2questions = []
        factlist = []
        questions = []
        
        for idx in range(0, len(titles), config.concurrent_api_requests_num):
            batch_titles = titles[idx:idx + config.concurrent_api_requests_num]
            batch_prompts = []
            
            for title in batch_titles:
                prompt = buildMessages(
                    SystemMessage(self.extraction_prompt.format(
                        main_theme=config.main_theme
                    )),
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
                
                extractions = parse_response(extractions_text,"关键[\d ]::+")
                if not extractions:
                    extractions = [extractions_text]
                extractions = [e for e in extractions if any(theme in e for theme in (main_theme_set + titleset))]
                for ext_idx in range(0, len(extractions), config.concurrent_api_requests_num):
                    batch_extractions = extractions[ext_idx:ext_idx + config.concurrent_api_requests_num]
                    question_prompts = []
                    
                    for extraction in batch_extractions:
                        question_prompt = DEFAULT_QUESTION_PROMPT + self.question_prompt if self.question_prompt else DEFAULT_QUESTION_PROMPT
                        
                        prompt = buildMessages(
                            UserMessage(question_prompt.format(extraction=extraction,num_questions_per_title=self.num_questions_per_title))
                        )
                        question_prompts.append(prompt)
                        
                    batch_gen_questions = await self.api.async_chat(question_prompts,temperature=config.temperature)
                    
                    for extraction, gen_questions in zip(batch_extractions, batch_gen_questions):
                        logger.debug(f"{'-'*20}gen questions{'-'*20}")
                        logger.debug(f"{gen_questions}")
                        gen_questions = parse_response(gen_questions,"问题[\d\ ]+::")
                        logger.debug(f"{'-'*20}gen questions after parsed{'-'*20}")
                        logger.debug(f"{gen_questions}")

                        valid_questions = await self._validate_questions(gen_questions, config.concurrent_api_requests_num)
                        logger.debug(f"{'-'*20}Extraction{'-'*20}")
                        logger.debug(f"{extraction}")
                        logger.debug(f"{'-'*20}Generated Questions{'-'*20}")
                        for q in gen_questions:
                            logger.debug(q)
                        if len(valid_questions) > 0:
                            extraction2questions.append({
                                "extraction": extraction,
                                "questions": valid_questions
                            })
                            questions.extend(valid_questions)
                        logger.debug(f"{'-'*20}Valid Questions{'-'*20}")
                        logger.debug(valid_questions)
        
        logger.error("end end end")
        return questions, factlist, extraction2questions

    async def _validate_questions(self, questions: List[str], concurrent_requests: int) -> List[str]:
        valid_questions = []
        for q_idx in range(0, len(questions), concurrent_requests):
            batch_questions = questions[q_idx:q_idx + concurrent_requests]
            validation_prompts = []
            
            for q in batch_questions:
                prompt = buildMessages(
                    UserMessage(f"""请判断下列问题是否是无效提问，无效提问的特征如下：
1.非疑问句，包含提问以外的答案、回答、转述原文、错误信息、自言自语、道歉等无意义信息。
2.问题逻辑不通顺，提问方式不自然, 自相矛盾。出现了"文本","根据文本"等字样。
3.提问风格、提问重点或表达方式奇怪，与人类习惯有明显差异。
4.指代不明，问题中包含了指向不明的代词，如"这个"、"那个"、"它"、"本次"、"今天"等。
具有以上任一特征的都会被视为无效提问。
问题:{q}
请先给出简要的打分理由，然后在最后一行输出判断'【无效】'或'【有效】'""")
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
                    SystemMessage(answer_prompt+ """注意：如果问题指代不明，例如包含('这','他','那次'等)代词，或无法从文本获取答案，则输出"无法回答"。"""
                                  + self.answer_prompt if self.answer_prompt else DEFAULT_ANSWER_PROMPT+ """注意：如果问题指代不明，例如包含('这','他','那次'等)代词，或无法从文本获取答案，则输出"无法回答"。"""
                                  ),
                    UserMessage(f"文本：{text}\n问题：{question}" + self.answer_prompt if self.answer_prompt else f"文本：{text}\n问题：{question}")
                )
                logger.info(prompt)
                prompts.append(prompt)
                
            batch_answers = await self.api.async_chat(prompts,temperature=self.config.temperature)
            
            for q, a in zip(batch_questions, batch_answers):
                logger.debug(f"{'-'*20}QA pair{'-'*20}")
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
                    SystemMessage("你是一个擅长阅读文本，回答人类问题的AI助手,请回答以下用户提问。"),
                    UserMessage(f"""文本：{text}
问题：{q}
请根据所给文本高质量地回答上述问题，回答应正确、通顺、清晰，据有深度。不应出现"根据文本","文本中"等字眼。""" + self.answer_prompt)
                )
                prompts.append(prompt)
            batch_answers = await self.api.async_chat(prompts,temperature=self.config.temperature)
            for q, a in zip(batch_questions, batch_answers):
                logger.debug(f"{'-'*20}rewritten QA pair{'-'*20}")
                logger.debug(f"{'-'*15}Question{'-'*15}")
                logger.debug(f"{q}")
                logger.debug(f"{'-'*15}Answer{'-'*15}")
                logger.debug(f"{a}")
            new_answers.extend(batch_answers)
        return new_answers

class backtranslation_rewrite(Strategy):
    def _create_text_retriever(self) -> TextRetriever:
        return BaseTextRetriever(self.api)
    
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

