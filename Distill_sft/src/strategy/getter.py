from strategy.genQA import genQA
from strategy.basic import BasicQA
from strategy.visGen import VisGenQA
from strategy.backtranslation_rewrite import backtranslation_rewrite
class StrategyGetter:
    @staticmethod
    def get_strategy(method_name):
        if method_name == 'basic':
            return BasicQA
        elif method_name == 'genQA':
            return genQA
        elif method_name == "backtranslation_rewrite":
            return backtranslation_rewrite
        elif method_name == "VisGen":
            return VisGenQA
        else:
            raise ValueError(f"Unknown method: {method_name}")