from typing import List
from .response_curation import EMStrategy,RougeStrategy
from .reasoning_curation import LengthFilterStrategy
from .judge import LLMAsJudgeStrategy

class Curator:
    def __init__(self, distill_args, clients):
        self.strategies = []
        self.distill_args = distill_args
        self.clients = clients
        self.chat_client = clients["chat"]
        if distill_args.llm_as_judge:
            self.judge_client = clients["judge"]

        if distill_args.exact_match:
            self.strategies.append(EMStrategy())

        self.strategies.sort(key=lambda s: s.priority)
        if self.distill_args.llm_as_judge:
            self.strategies.append(LLMAsJudgeStrategy(self.judge_client))
        print("Strategy")
        print(self.strategies)

    async def curate(self, candidates:List[str],answer:str)->List[str]:
        for strategy in self.strategies:
            candidates = await strategy.apply(candidates,answer)
            if not candidates:
                break
        print("CANDIDATESSS")
        print(len(candidates))
        print(candidates)
        
        
        return candidates[0] if candidates else []
        # return candidates
