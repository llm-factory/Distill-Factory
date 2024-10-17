from method import *
from ..common.message import *
from ..tools.tool import *
from tqdm import tqdm

from loguru import logger
class genQA(Strategy):
    def __init__(self,api):
        super().__init__(api)
        
    
    async def run(self, text, main_theme,num_question_per_title=10,batch_size = 1):
        
        print('-----------titles----------------')
        titles = self.genTitle(text,main_theme)
        # personas = self.getPersona(text)
        questions = await self.generateQuestions(text,main_theme,num_question_per_title,titles,batch_size=batch_size)
        answers = await self.getAnswers(text,questions,batch_size=batch_size)
        return questions,answers

    def genTitle(self,text,main_theme):
        prompt = buildMessages(
            [
            UserMessage(f"{text}\n根据以上文本提取与{main_theme}相关的具有概括性的若干个小标题。小标题必须包含准确的信息，例如准确的时间、地点、人物、名称、事件等，不能有歧义，不能指向模糊。每个小标题一行，不要有重复.")
            ]
        )            
        
        titles = clean_and_split_multiline_reply(self.api.chat(prompt))
        print('--------titles------------')
        print(titles)
        return titles
    
    
    def getPersona(self, text):
        prompt = buildMessages(
            [
                UserMessage(
                    f"根据以下文本：\n{text}生成10个可能对该文本感兴趣的大致人物描述，不能包含人名，每个人物一行"
                )
            ]
        )
        personas = clean_and_split_multiline_reply(self.api.chat(prompt))
        return personas
        
    async def generateQuestions(self, text, main_theme, num_question_per_title,titles,batch_size=1):
        questions = []
        
        for idx in tqdm(range(0,len(titles),batch_size),desc='Generating questions'):
            batch_titles = titles[idx:idx+batch_size]
            prompts = []
            for i in range(len(batch_titles)):
                prompt = buildMessages(
                    [
                        UserMessage(
                            f"根据以下文本：\n" + text + f"指向{main_theme}生成{num_question_per_title}个在不同场景下与“{batch_titles[i]}”有关的问题，每个问题一行，以数字加”.“开始，不能重复"
                        )
                    ]
                )
                prompts.append(prompt)
            genQuestions = await self.api.async_chat(prompts)
            genQuestions = clean_and_split_multiline_replyList(genQuestions)
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
            genQuestions = clean_and_split_multiline_replyList(genQuestions)
        return questions
    
    
    async def getAnswers(self, text,questions,batch_size=1):
        answers = []
        for idx in tqdm(range(0,len(questions),batch_size),desc="Generating answers"):
            batch_questions = questions[idx:idx+batch_size]
            prompts=  []
            for i in range(len(batch_questions)):
                prompt = buildMessages(
                    [
                        UserMessage(
                            f"{text}\n根据以上文本直接回答以下问题：{batch_questions[i]} 答案中不能出现'根据文本','根据文本中'等字样"
                        )
                    ]
                )
                prompts.append(prompt)
            
            genAnswers = await self.api.async_chat(prompts)
            print('-----------genAnswers----------------')
            print(genAnswers)
            answers.extend(genAnswers)
    
        return answers