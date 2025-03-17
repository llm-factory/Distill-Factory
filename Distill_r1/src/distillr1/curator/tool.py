import re

def extract_boxed_answer(answer: str) -> str:
    matches = re.findall(r'\\boxed{([^}]*)}', answer)
    return matches