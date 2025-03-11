from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from strategy.method import TextRetriever, Generator, Verifier

class Strategy(ABC):
    def __init__(self, api,config):
        self.api = api
        self.config = config
        self.text_retriever = self._create_text_retriever()
        self.qa_generator = self._create_qa_generator()
        self.qa_verifier = self._create_qa_verifier()


    @abstractmethod
    def _create_text_retriever(self) -> TextRetriever:
        pass

    @abstractmethod
    def _create_qa_generator(self) -> Generator:
        pass

    @abstractmethod
    def _create_qa_verifier(self) -> Verifier:
        pass

    @abstractmethod
    async def run(self, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        pass