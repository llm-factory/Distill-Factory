import re
from typing import List
from .pattern import *;
import jieba
def questions_filter(questions:List[str])-> List[str]:
    filtered = []
    for q in questions:
        if not any([re.match(pattern, q) for pattern in ABANDONED_PATTERN_IN_QUESTIONS]):
            filtered.append(q)
    filtered = [question_deduplication(q) for q in filtered]
    return filtered

def question_deduplication(questions:List[str]):
    """
    TODO
    """
    return questions

def answers_filter(answers:List[str]):
    filtered = []
    idxs_to_remove = []
    for idx,a in enumerate(answers):
        if not any([re.match(pattern, a) for pattern in ABANDONED_PATTERN_IN_ANSWERS]):
            filtered.append(a)
        else:
            idxs_to_remove.append(idx)
    return filtered,idxs_to_remove