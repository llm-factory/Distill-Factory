import re
ABANDONED_PATTERN_IN_QUESTIONS = [
    r"^[\*\^\.\/\\\" ]*(回答|答案|解答|结果|答)",
    r"^[\*\^\.\/\\\" ]*([Aa]nswer|[Rr]eply|[Rr]esponse)",
]

ABANDONED_PATTERN_IN_ANSWERS = [
    r"(无法|不能|不可以|没办法|我不能|我不)(给出|提供|推断|判断)?(回答|确定|解答)",
]