from transformers import GPT2LMHeadModel

class ModelCheckpoint:
    def __init__(self, filepath:str, mode:str='min'):
        self.mode = mode
        self.best_score = None
        self.filepath = filepath
        self.is_better = self._get_comparator(mode)

    def _get_comparator(self, mode:str):
        if mode == 'min':
            return lambda current: current < self.best_score
        elif mode == 'max':
            return lambda current: current > self.best_score
        else:
            raise ValueError("Mode should be 'min' or 'max'.")

    def __call__(self, current_score:float, model:GPT2LMHeadModel):
        if self.best_score is None or self.is_better(current_score):
            self.best_score = current_score
            model.save_pretrained(self.filepath)