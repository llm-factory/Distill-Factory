from abc import ABC, abstractmethod
from typing import List
class CurationStrategy(ABC):
    def __init__(self, priority):
        self.priority = priority
    @abstractmethod
    async def apply(self, candidates)->List[str]:
        pass
