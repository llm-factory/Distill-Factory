
from ..curator.curator import Curator
from ..api.protocol import ChatMessage,UserMessage,AssistantMessage
from ..api.misc import build_messages
from ..utils.tool import save_to_json
from typing import Tuple,List
import asyncio
class Distiller():
    def __init__(self,distill_args,clients,questions,answers):
        self.distill_args = distill_args
        self.clients = clients
        self.questions = questions
        self.answers = answers
        self.chat_client = clients["chat"]
        self.curator = Curator(distill_args,clients)
        if distill_args.llm_as_judge:
            self.judge_client = clients["judge"]
        
    async def distill(self):
        batch_size = 4 # TODO
        all_questions,all_curated_output = [],[]        
        
        for i in range(0,len(self.questions),batch_size):
            batch_questions = self.questions[i:i+batch_size]
            batch_answers = self.answers[i:i+batch_size]
            batch_tasks = [self.distill_single(question,answer) for question,answer in zip(batch_questions,batch_answers)]
            results = await asyncio.gather(*batch_tasks)
            batch_questions = [item[0] for item in results]
            batch_curated_output = [item[1] for item in results]            
            all_questions.extend(batch_questions)
            all_curated_output.extend(batch_curated_output)
            save_to_json(all_questions,all_curated_output,self.distill_args.output_path)
        
    async def distill_single(self,question:str,answer:str)->Tuple[str,str]:
        prompts = [
            [
                UserMessage(self.distill_args.meta_prompt + question)
            ] for i in range(self.distill_args.roll_out_size)
        ]
        if self.distill_args.roll_out_size > 1 and self.chat_client.get_generating_args().get("temperature", None) == 0:
            raise ValueError("Performing roll out with temperature=0, Increase temperature or set roll_out_size to 1")
        
        candidates = await self.chat_client.async_chat(prompts)

        curated_output = await self.curator.curate(candidates,answer)
        
        return question,curated_output