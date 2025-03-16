from abc import ABC, abstractmethod

class Curator(ABC):
    def __init__(self, distill_args, clients):
        self.distill_args = distill_args
        self.chat_client = clients["chat"]
        self.reward_client = clients["reward"]
    
    @abstractmethod
    def curate(self):
        pass
