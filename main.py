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
model_config_dir = os.getenv("MODEL_CONFIG_DIR")
tokenizer_config_dir = os.getenv("TOKENIZER_CONFIG_DIR")
model_dir = os.getenv("PRETRAINED_MODEL_DIR")
tokenizer_dir = os.getenv("PRETRAINED_TOKENIZER_DIR")

model_config = Config(model_config_dir).load()
tokenizer_config = Config(tokenizer_config_dir).load()
generator = Generator().manual_seed(model_config["random_seed"])

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.manual_seed_all(model_config["random_seed"])

tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint, 
                                              device_map=device, 
                                              bos_token=tokenizer_config["bos_token"], 
                                              eos_token=tokenizer_config["eos_token"], 
                                              unk_token=tokenizer_config["unk_token"], 
                                              pad_token=tokenizer_config["pad_token"])

tokenizer.save_pretrained(tokenizer_dir)

model = GPT2LMHeadModel.from_pretrained(checkpoint, device_map=device)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
optimizer = AdamW(model.parameters(), lr=model_config["learning_rate"], eps=model_config["epsilon"])

dataset = RecipeDataset(data_path, tokenizer, tokenizer_config["seq_length"], device)
train_dataset, val_dataset = dataset.split(val_p=model_config["val_p"], generator=generator)

train_sampler = RandomSampler(train_dataset, replacement=True, generator=generator)
val_sampler = RandomSampler(val_dataset, replacement=True, generator=generator)

train_loader = DataLoader(train_dataset, batch_size=model_config["batch_size"], sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=model_config["batch_size"], sampler=val_sampler)

trainer = Trainer(optimizer, model_dir, model_config["logs_dir"])
trainer.train(model, train_loader, val_loader, model_config["epochs"])