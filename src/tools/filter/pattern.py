import re
ABANDONED_PATTERN_IN_QUESTIONS = [
    r"^(回答|答案|解答|结果|答)",
    r"^([Aa]nswer|[Rr]eply|[Rr]esponse)",
]

ABANDONED_PATTERN_IN_ANSWERS = [
    r"无法(回答|确定|解答)"
]