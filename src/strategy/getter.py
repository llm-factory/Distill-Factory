from strategy.genQA import genQA
class StrategyGetter:
    @staticmethod
    def get_strategy(method_name):
        if method_name == 'genQA':
            return genQA
        else:
            raise ValueError(f"Unknown method: {method_name}")