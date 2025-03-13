from typing import List
from ..api.client import Client
from rouge import Rouge

SYSTEM_JUDGE_PROMPT = """
You are a judge that evaluates the correctness of a solution.
You are given a question, an answer and a ground truth answer.
You need to judge whether the answer is correct or not.

Question: {Question}
Answer: {Answer}
Ground Truth Answer: {GroundTruthAnswer}

Please judge whether the answer is correct or not.
Please only output \\boxed{{}} in the format of \\boxed{{}}.
"""


class AnswerJudger():
    def __init__(self, judge_mode: List[str], rouge_threshold=0, base_url='', api_key='', model=''):  # llm, em, rouge
        self.judge_mode = judge_mode

        if 'llm' in self.judge_mode:
            self.client = Client(base_url, api_key)
            self.model = model

        if 'rouge' in self.judge_mode:
            self.rouge = Rouge()
            self.rouge_threshold = rouge_threshold

    async def judge(self, question: str, answer: str, llm_answer: str) -> bool:
        if 'llm' in self.judge_mode:
            if not await self.judge_answer_llm(question, answer, llm_answer):
                return False

        if 'em' in self.judge_mode:
            if not self.judge_answer_exact_match(answer, llm_answer):
                return False

        if 'rouge' in self.judge_mode:
            if not self.judge_answer_rouge(answer, llm_answer):
                return False

        return True

    async def judge_answer_llm(self, question: str, answer: str, llm_answer: str):
        judge_result = await self.client.judge_answer_correctness(self.model, question, answer, llm_answer)
        return judge_result

    def judge_answer_exact_match(self, answer: str, llm_answer: str) -> bool:
        if answer == llm_answer:
            return True
        return False

    def judge_answer_rouge(self, answer: str, llm_answer: str) -> bool:
        score = self.rouge.get_scores(answer, llm_answer)[0]['rouge-l']['f']
        if score > self.rouge_threshold:
            return True
        return False
