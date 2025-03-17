from ..strategy import CurationStrategy
from rouge import Rouge

class RougeStrategy(CurationStrategy):
    def __init__(self, reference_answer, rouge_threshold=0.5):
        self.reference_answer = reference_answer
        self.rouge_threshold = rouge_threshold
        self.rouge = Rouge()

    async def apply(self, candidates):
        filtered_candidates = []
        for candidate in candidates:
            if self.judge_answer_rouge(self.reference_answer, candidate.response):
                filtered_candidates.append(candidate)
        return filtered_candidates

    def judge_answer_rouge(self, answer: str, llm_answer: str) -> bool:
        score = self.rouge.get_scores(answer, llm_answer)[0]['rouge-l']['f']
        return score > self.rouge_threshold
