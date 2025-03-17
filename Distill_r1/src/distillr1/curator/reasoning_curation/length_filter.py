from ..strategy import CurationStrategy

class LengthFilterStrategy(CurationStrategy):
    def __init__(self, min_length,max_length):
        self.priority = 2
        self.min_length = min_length
        self.max_length = max_length

    async def apply(self, candidates):
        return [c for c in candidates if len(c) > self.min_length]