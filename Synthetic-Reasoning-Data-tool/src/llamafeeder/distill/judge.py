SYSTEM_JUDGE_PROMPT= """
You are a judge that evaluates the correctness of a solution.
You are given a question, an answer and a ground truth answer.
You need to judge whether the answer is correct or not.

Question: {question}
Answer: {llm_answer}
Ground Truth Answer: {answer}

Please judge whether the answer is correct or not.
Please only output \\boxed{{}} in the format of \\boxed{{}}.
"""