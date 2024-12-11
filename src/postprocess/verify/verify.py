from common.message import buildMessages, SystemMessage, UserMessage
from typing import List, Tuple
import logging

class Verifier:
    def __init__(self, api):
        self.api = api

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