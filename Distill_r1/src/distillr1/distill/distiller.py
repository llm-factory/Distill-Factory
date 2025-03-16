
from .curator import Curator
from ..api.protocol import ChatMessage,UserMessage,AssistantMessage
from ..api.misc import build_messages
from ..utils.tool import save_to_json
class Distiller():
    def __init__(self,distill_args,clients,questions,answers):
        self.distill_args = distill_args
        self.clients = clients
        self.questions = questions
        self.answers = answers
        self.chat_client = clients["chat"]
        self.curator = Curator(distill_args,clients)
        if distill_args.enable_reward_model:
            self.reward_client = clients["reward"]
        
    async def distill(self):
        
        for question, answer in zip(self.questions, self.answers):
            question = question[0]['content']
            answer = answer[0]['content']
            prompts = [
                [
                    build_messages(
                        UserMessage(self.distill_args.meta_prompt + question)
                    )
                ] * self.distill_args.roll_out_size
            ]
            if self.distill_args.roll_out_size > 1 and self.chat_client.get_generating_args().get("temperature", None) == 0:
                raise ValueError("Performing roll out with temperature=0, Increase temperature or set roll_out_size to 1")
            
            candidates = await self.chat_client.async_chat(prompts)
            curated = await self.curator.curate(candidates)
            save_to_json(curated,self.distill_args.output_path)