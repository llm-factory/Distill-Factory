from strategy.method import Strategy
from common.message import *
from tools.tool import clean_and_split_reply,clean_and_split_reply_list,clean_and_split_titles,clean_and_split_title_list
from tools.filter import *
from tqdm import tqdm

from loguru import logger
class genQA(Strategy):
    def __init__(self,api):
        super().__init__(api)
        
    
    async def run(self, text, main_theme,num_question_per_title=10,concurrent_api_requests_num = 1):
        
        titles = self.genTitle(text,main_theme)
        titles = await self.splitTitles(titles,concurrent_api_requests_num)
        print('-----------titles----------------')
        print(titles)
        questions = await self.generateQuestions(text,main_theme,num_question_per_title,titles,concurrent_api_requests_num=concurrent_api_requests_num)
        questions = questions_filter(questions)
        answers = await self.getAnswers(text,questions,concurrent_api_requests_num=concurrent_api_requests_num)
        answers,idxs_to_remove = answers_filter(answers)
        questions = [q for idx,q in enumerate(questions) if idx not in idxs_to_remove]
        return questions,answers


    def genTitle(self,text,main_theme):
        prompt = buildMessages(
            [
            UserMessage(f"{text}\n根据以上文本提取与{main_theme}相关的具有概括性的若干个小标题。小标题必须包含准确的信息，例如准确的时间、地点、人物、名称、事件等，不能有歧义，不能指向模糊。每个小标题一行，不要有重复.")
            ]
        )            
        
        titles = clean_and_split_titles(self.api.chat(prompt))
        print('--------titles------------')
        print(titles)
        return titles
    
    async def splitTitles(self,titles,concurrent_api_requests_num=1):
        splitTitles = []
        for idx in tqdm(range(0,len(titles),concurrent_api_requests_num),desc='Splitting titles'):
            batch_titles = titles[idx:idx+concurrent_api_requests_num]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                    [
                        SystemMessage(
                        f"你是一个擅长划分标题的助手。以下是一个标题,该标题可能包含多个事实，不够简洁。对于包含多个事实的标题你需要将该标题划分为多个小标题,每个小标题要包含原标题中的核心信息和一部分有效信息，不能改变原意，每个小标题一行。对于已经足够简洁的标题，则输出原标题。"    
                        ),
                        UserMessage(
                            f"""标题: {batch_titles[i]}\n 只输出划分后的小标题或者原标题"""
                        )
                    ]
                )
                prompts.append(prompt)
            titles = await self.api.async_chat(prompts)
            titles = clean_and_split_title_list(titles)
            print('-----------splitTitles----------------')
            print(titles)
            splitTitles.extend(titles)
        splitTitles = list(set(splitTitles))
        return splitTitles
    
        
    def getPersona(self, text):
        prompt = buildMessages(
            [
                UserMessage(
                    f"根据以下文本：\n{text}生成10个可能对该文本感兴趣的大致人物描述，不能包含人名，每个人物一行"
                )
            ]
        )
        personas = clean_and_split_reply(self.api.chat(prompt))
        return personas
        
    async def generateQuestions(self, text, main_theme, num_question_per_title,titles,concurrent_api_requests_num=1):
        questions = []
        
        for idx in tqdm(range(0,len(titles),concurrent_api_requests_num),desc='Generating questions'):
            batch_titles = titles[idx:idx+concurrent_api_requests_num]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                    [
                        UserMessage(
                            f"根据以下文本：\n" + text + f"指向{main_theme}生成{num_question_per_title}个在不同场景下与“{batch_titles[i]}”有关的可以根据文本内容回答的问题，问题必须明确指向{main_theme}以避免指向模糊。每个问题一行，以数字加”.“开始，不能重复"
                        )
                    ]
                )
                prompts.append(prompt)
            genQuestions = await self.api.async_chat(prompts)
            genQuestions = clean_and_split_reply_list(genQuestions)
            print('-----------genQuestions----------------')
            print(genQuestions)
            questions.extend(genQuestions)
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                    [
                        UserMessage(
                            f"根据以下文本：\n{text}指向{main_theme}生成{num_question_per_title}个在不同场景下不含“{i}”但需要结合文本中关于{batch_titles[i]}的知识才能回答的问题，每个问题一行，以数字加”.“开始，不能重复"
                        )
                    ]
                )
                prompts.append(prompt)
            genQuestions = await self.api.async_chat(prompts)
            genQuestions = clean_and_split_reply_list(genQuestions)
            questions.extend(genQuestions)
        return questions
    
    
    async def getAnswers(self, text,questions,concurrent_api_requests_num=1):
        answers = []
        for idx in tqdm(range(0,len(questions),concurrent_api_requests_num),desc="Generating answers"):
            batch_questions = questions[idx:idx+concurrent_api_requests_num]
            prompts=  []
            for i in range(len(batch_questions)):
                prompt = buildMessages(
                    [
                        UserMessage(
                            f"{text}\n根据以上文本直接回答以下问题：{batch_questions[i]} 答案中不能出现'根据文本','根据文本中'等字样。对于无法从文本中获取答案的问题，输出“无法回答”。"
                        )
                    ]
                )
                prompts.append(prompt)
            
            genAnswers = await self.api.async_chat(prompts)
            print('-----------genAnswers----------------')
            print(genAnswers)
            answers.extend(genAnswers)
        return answers