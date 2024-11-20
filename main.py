import os
import torch
from config import Config
from trainer import Trainer
from torch import Generator
from torch.optim import AdamW
from dotenv import load_dotenv
from data_utils import RecipeDataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

load_dotenv()
data_path = os.getenv("DATA_DIR")
checkpoint = os.getenv("MODEL_CHECKPOINT")
train_config_dir = os.getenv("MODEL_CONFIG_DIR")
model_dir = os.getenv("PRETRAINED_MODEL_DIR")
tokenizer_dir = os.getenv("PRETRAINED_TOKENIZER_DIR")

config = Config(train_config_dir)
train_config = config.load()
generator = Generator().manual_seed(train_config["random_seed"])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(train_config["random_seed"])

tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint, 
                                              device_map=device, 
                                              bos_token=train_config["bos_token"], 
                                              eos_token=train_config["eos_token"], 
                                              unk_token=train_config["unk_token"], 
                                              pad_token=train_config["pad_token"])

tokenizer.save_pretrained(tokenizer_dir)

model = GPT2LMHeadModel.from_pretrained(checkpoint, device_map=device)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
optimizer = AdamW(model.parameters(), lr=train_config["learning_rate"], eps=train_config["epsilon"])

dataset = RecipeDataset(data_path, tokenizer, train_config["seq_length"], device)
train_dataset, val_dataset = dataset.split(val_p=train_config["val_p"], generator=generator)

train_sampler = RandomSampler(train_dataset, replacement=True, generator=generator)
val_sampler = RandomSampler(val_dataset, replacement=True, generator=generator)

train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], sampler=val_sampler)

trainer = Trainer(optimizer, model_dir, train_config["logs_dir"])
trainer.train(model, train_loader, val_loader, train_config["epochs"])