import pandas as pd
from torch import Generator
from typing import Tuple, List
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, random_split

class RecipeDataset(Dataset):
    def __init__(self, data_dir:str, tokenizer:GPT2TokenizerFast, seq_length:int, device:str):
        self._data = None
        self.device = device
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = pd.read_csv(data_dir)
            
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, df:pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self._data = df.values.squeeze(axis=1)
    
    def _tokenize(self, sequence):
        inputs = self.tokenizer(sequence, truncation=True, padding="max_length", max_length=self.seq_length, return_tensors="pt").to(device=self.device)
        inputs["labels"] = inputs.input_ids.clone()
        return inputs
    
    def split(self, val_p:float, generator:Generator) -> Tuple[List, List]:
        val_size = int(val_p * len(self))
        train_size = len(self) - val_size
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            sequences = self.data[key]
            inputs = [self._tokenize(sequence) for sequence in sequences]
        else:
            sequence = self.data[key]
            inputs = self._tokenize(sequence)
        return inputs
    
    def __len__(self):
        return len(self.data)