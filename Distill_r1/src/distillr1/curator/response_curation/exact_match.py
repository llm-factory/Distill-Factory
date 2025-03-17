from ..strategy import CurationStrategy
from ..tool import extract_boxed_answer
from math_verify import parse, verify

class EMStrategy(CurationStrategy):
    def __init__(self, answer):
        self.priority=1
        self.answer = answer

    async def apply(self, candidates):
        filtered_candidates = []
        
        for candidate in candidates:
            candidate_answer_list = extract_boxed_answer(candidate.response)
            gold_answer_list = self.answer.split(';')
            if len(candidate_answer_list) != len(gold_answer_list):
                continue
            all_correct = True 
            for candidate_answer_ele, gold_answer_ele in zip(candidate_answer_list, gold_answer_list):
                parsed_candidate_answer = parse(candidate_answer_ele)
                parsed_gold_answer = parse(gold_answer_ele)
                if not verify(parsed_candidate_answer, parsed_gold_answer):
                    all_correct = False
                    break            
            if all_correct:
                filtered_candidates.append(candidate)
        return filtered_candidates
