from strategy.method import Strategy
from common.message import *
from tools.tool import *
from tools.filter import *
from tools.tool import read_file
from tqdm import tqdm
from io import StringIO
import asyncio
import logging
import random

logger = logging.getLogger('logger')

class genQA(Strategy):
    def __init__(self,api):
        super().__init__(api)
    
    async def process_single_file(self, config,file_path: str, num_question_per_title: int, concurrent_api_requests_num: int):
        main_theme = config.main_theme
        datas = load_datas(file_path,config) 
        questions = []
        answers = []
        for i in range(0,len(datas),concurrent_api_requests_num):
            batch_datas = datas[i:i+concurrent_api_requests_num]
            tasks = []
            for data in batch_datas:
                texts= parse_data(data,config)
                for text in texts:
                    if(text == ""):
                        continue
                    task = self.process_single_data(config,text,file_path,main_theme,num_question_per_title,concurrent_api_requests_num)
                    tasks.append(task)
            results = await asyncio.gather(*tasks)
            for question, answer in results:
                questions.extend(question)
                answers.extend(answer)
        return questions, answers
    
    async def process_single_data(self, config,text,file_path: str, main_theme: str, num_question_per_title: int, concurrent_api_requests_num: int,additional_info=""):
        logger.debug('='*30 + f'Text of {file_path}' + '='*30)
        logger.debug(text[:200])
        logger.debug(f"{'=' * 30}Generating Titles For {file_path}{'=' * 30}")
        genTitles = await self.genTitle(text, main_theme)
        logger.debug(f"{'=' * 30}Titles of {file_path}{'='*30}")
        logger.debug(genTitles)
        logger.debug(f"{'=' * 30}Splitting Titles For {file_path}{'=' * 30}")
        titles = await self.splitTitles(genTitles,concurrent_api_requests_num)
        logger.debug(f"{'=' * 30}Titles of {file_path}{'='*30}")
        logger.debug(titles)
        logger.debug(f"{'=' * 30}Generating Questions For {file_path}{'='*30}")
        questions = await self.generateQuestions(text, main_theme, num_question_per_title, titles, concurrent_api_requests_num=concurrent_api_requests_num,additional_info=additional_info)
        questions = questions_filter(questions)
        logger.debug(questions)
        logger.debug(f"{'=' * 30}Generating Answers For {file_path}{'='*30}")
        answers = await self.getAnswers(text, questions, concurrent_api_requests_num=concurrent_api_requests_num,main_theme=main_theme)
        answers, idxs_to_remove = answers_filter(answers)
        questions = [q for idx, q in enumerate(questions) if idx not in idxs_to_remove]
        logger.debug(f"{'=' * 30}verifying QAs of {file_path}{'='*30}")
        questions,answers = await self.verifyQA(text,main_theme,questions,answers,concurrent_api_requests_num)
        save_QA_dataset(questions,answers,config.save_dir,config.save_file_name)
        return questions, answers
    
    async def verifyQA(self,text,main_theme,questions,answers,concurrent_api_requests_num=1):
        prompts = []
        new_questions = []
        new_answers = []
        for i in range(0,len(questions),concurrent_api_requests_num):
            batch_questions = questions[i:i+concurrent_api_requests_num]
            batch_answers = answers[i:i+concurrent_api_requests_num]
            prompts = []
            for q,a in zip(batch_questions,batch_answers):
                prompt = buildMessages(
                        SystemMessage(
                            "你需要根据提供的文本，判断给定的问答是否“有效”。\n"
                            "有效的问答的标准是：\n"
                            f"问题合理。问题与文本主题 {main_theme} 相关，逻辑清晰，不混乱，符合人类习惯。问题中不包含回答等无关信息。\n" 
                            "问题完整，不含有省略、不完整、或出现意外截断情况的问题。\n"
                            "回答正确，完整。答案可由文本信息支持，与问题所问内容相符，不包含'根据文本内容'等字眼。回答逻辑清晰，不包含无关信息。\n"
                            "若符合以上所有标准，则回答“有效”；如果任一条件不满足，则回答“无效”。\n"
                            "请先简述你判断的原因，然后在最后一行输出'有效'或'无效'，不得输出其他信息。"
                        ),
                        UserMessage(
                            f"{text}\n\n请根据上面的文本判断以下问答是否有效：\n"
                            f"问题: {q}\n"
                            f"答案: {a}\n"
                            "请先简述你判断的原因，然后在最后一行输出'有效'或'无效'，不得输出其他信息。"
                        )
                    )
                prompts.append(prompt)
            replies = await self.api.async_chat(prompts)
            bin = [0 if "无效" in r.split()[-1] + r.split()[0] else 1 for r in replies]
            delete_num = len([idx for idx in bin if idx == 0])
            verified_Q = [q for idx,q in enumerate(batch_questions) if bin[idx] == 1]
            verified_A = [a for idx,a in enumerate(batch_answers) if bin[idx] == 1]
            new_questions.extend(verified_Q)
            new_answers.extend(verified_A)
        return new_questions,new_answers
    
    
    async def run(self, config, num_question_per_title=3, concurrent_api_requests_num=1):
        init_QA_dataset(config.save_dir,config.save_file_name)
        all_questions = []
        all_answers = []
        file_paths = getFilePaths(config.file_folder,config.file_path,config.file_type)
        logger.debug(f"{'=' * 30}File Paths{'='*30}")
        logger.debug(file_paths)
        tasks = [
            self.process_single_file(
                config,
                file_path,
                num_question_per_title, 
                concurrent_api_requests_num
            )
            for file_path in file_paths
        ]
        if(config.is_structure_data):    
            concurrent_api_requests_num = 1
        for i in range(0,len(tasks),concurrent_api_requests_num):
            batch_tasks = tasks[i:i+concurrent_api_requests_num]
            results = await asyncio.gather(*batch_tasks)
                    
            for questions, answers in results:
                all_questions.extend(questions)
                all_answers.extend(answers)
            
        return all_questions, all_answers

    async def genTitle(self,text,main_theme):
        prompt = buildMessages(
            SystemMessage(
                "你是一个优秀的文本阅读与信息提炼助手。请根据下面提供的文本内容，提取出多个精准且具有针对性的小标题。这些小标题应满足以下要求：\n"
                "1. 每个小标题必须包含确切的时间（若文本中有提及），地点，人物名称，事件名称或组织名称等具体信息。\n"
                "2. 小标题必须清晰明了，不含模糊表达或歧义性描述。\n"
                "3. 小标题应从整体上覆盖文本中的核心主题及重要信息点，确保不遗漏关键事件。\n"
                "4. 每个小标题独占一行，不能重复\n"
                "5. 直接输出小标题，不要有其他信息。"
            ),
            UserMessage(f"{text}")
        )            
        titles = await self.api.async_chat([prompt])
        titles = clean_and_split_title_list(titles)
        return titles
    
    async def splitTitles(self,titles,concurrent_api_requests_num=1):
        splitTitles = []
        for idx in tqdm(range(0,len(titles),concurrent_api_requests_num),desc='Splitting titles'):
            batch_titles = titles[idx:idx+concurrent_api_requests_num]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                        SystemMessage(
                        f"你是一个擅长划分标题的助手。以下是一个标题,该标题可能包含多个事实，不够简洁。对于包含多个事实的标题你需要将该标题划分为多个小标题，不能改变原意，每个小标题一行,不输出额外信息。对于已经足够简洁的标题，则输出原标题。"    
                        ),
                        UserMessage(
                            f"""标题: {batch_titles[i]}\n 只输出划分后的小标题或者原标题，不要有其他信息。"""
                        )
                )
                prompts.append(prompt)
            split_title =  await self.api.async_chat(prompts)
            split_title = clean_and_split_title_list(split_title)
            splitTitles.extend(split_title)
        splitTitles = list(set(splitTitles))
        return splitTitles
    
        
    def getPersona(self, text):
        prompt = buildMessages(
                UserMessage(
                    f"根据以下文本：\n{text}生成10个可能对该文本感兴趣的大致人物描述，不能包含人名，每个人物一行"
                )
        )
        personas = clean_and_split_reply(self.api.chat(prompt))
        return personas
        
    async def generateQuestions(self, text, main_theme, num_question_per_title,titles,concurrent_api_requests_num=1,additional_info=""):
        questions = []
        for idx in tqdm(range(0,len(titles),concurrent_api_requests_num),desc='Generating questions'):
            batch_titles = titles[idx:idx+concurrent_api_requests_num]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                        SystemMessage(
                            f"你对{main_theme}相关内容十分感兴趣。请您根据以下文本内容，围绕'{main_theme}'提出{num_question_per_title}个清晰、客观的问题，"
                            f"这些问题必须紧密围绕'{batch_titles[i]}'，且可以在文本中找到明确的答案。\n"
                            f"你的问题需要满足以下要求：\n"
                            f"1. 问题必须包括完整的信息以避免模糊，例如：具体的人物、名称、事件、时间等。\n"
                            f"2. 问题应当客观、具体，避免模糊不清。"
                            f"3. 问题需基于客观事实，不得包含主观感受、预测或想象。问题应当能在文本中找到答案\n"
                            f"4. 确保问题内容不重复，包含不同类型的问题并且覆盖文本的不同部分或不同维度。\n"
                            f"每个问题以'问题'加数字加'::'开头，且问题内容不能重复。"
                            
                        ),
                        UserMessage(
                            f"文本:{text}\n每个问题以'问题'加数字加'::'开头，且问题内容不能重复。请提问。"
                        ),                        
                )
                prompts.append(prompt)
            genQuestions = await self.api.async_chat(prompts)
            logger.debug(f"{'-' * 20}Questions of {batch_titles}{'-'*20}")
            logger.debug(genQuestions)
            genQuestions = clean_and_split_question_list(genQuestions)
            questions.extend(genQuestions)
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                        SystemMessage(
                            f"你对{main_theme}相关内容十分感兴趣。请您根据以下文本内容，围绕'{main_theme}'提出{num_question_per_title}个清晰、客观的问题，"
                            f"这些问题中不能含'{batch_titles[i]}'，但能够从文本中与'{batch_titles[i]}'相关的信息回答\n"
                            f"你的问题需要满足以下要求：\n"
                            f"1. 问题必须包括完整的信息以避免模糊，例如：具体的人物、名称、事件、时间等。\n"
                            f"2. 问题应当客观、具体，避免模糊不清。"
                            f"3. 问题需基于客观事实，不得包含主观感受、预测或想象。问题应当能在文本中找到答案\n"
                            f"4. 确保问题内容不重复，包含不同类型的问题并且覆盖文本的不同部分或不同维度。\n"
                            f"每个问题以'问题'加数字加'::'开头，且问题内容不能重复。"
                        ),
                        UserMessage(
                            f"文本:{text}\n每个问题以'问题'加数字加'::'开头，且问题内容不能重复。请提问。"
                        ),                        
                )
                prompts.append(prompt)
            genQuestions = await self.api.async_chat(prompts)
            logger.debug(f"{'-' * 20}Questions of 2 {batch_titles}{'-'*20}")
            logger.debug(genQuestions)
            genQuestions = clean_and_split_question_list(genQuestions)
            questions.extend(genQuestions)
        return questions
    
    async def getAnswers(self, text,questions,concurrent_api_requests_num=1,main_theme=""):
        logger.debug("======================Generating Answers======================")
        answers = []
        for idx in tqdm(range(0,len(questions),concurrent_api_requests_num),desc="Generating answers"):
            batch_questions = questions[idx:idx+concurrent_api_requests_num]
            prompts=  []
            for i in range(len(batch_questions)):
                prompt = buildMessages(
                        SystemMessage(f"{text}\n你是一位AI助手。请从热情、耐心、专业、简洁中选择一个回答风格并礼貌地回答以下问题。对于无法回答的问题，请回复'无法回答'"),
                        UserMessage(
                            f"{batch_questions[i]}"
                        )
                )
                prompts.append(prompt)
            
            genAnswers = await self.api.async_chat(prompts)
            for q,a in zip(batch_questions,genAnswers):
                logger.debug(f"{'-'*20}QA pair{'-'*20}")
                logger.debug(f"{'-'*15}Question{'-'*15}")
                logger.debug(f"{q}")
                logger.debug(f"{'-'*15}Answer{'-'*15}")
                logger.debug(f"{a}")
            answers.extend(genAnswers)
        return answers