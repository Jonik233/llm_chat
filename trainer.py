import torch
from tqdm.auto import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from callbacks import ModelCheckpoint
from transformers import GPT2LMHeadModel
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    
    def __init__(self, optimizer:Optimizer, model_dir:str, logs_dir:str, lr_scheduler:LRScheduler=None):
        self.logs_dir = logs_dir
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpoint = ModelCheckpoint(model_dir)
        
        
    @torch.no_grad()
    def _val_model(self, model:GPT2LMHeadModel, val_loader:DataLoader) -> float:
        losses = []
        model.eval()
        for inputs in val_loader:
            outputs = model(**inputs)
            loss = outputs.loss
            losses.append(loss.item())

        loss = sum(losses) / len(losses)
        return loss


    def _train_model(self, model:GPT2LMHeadModel, train_loader:DataLoader) -> float:
        losses = []
        model.train()
        train_steps = len(train_loader)
        progress_bar = tqdm(range(train_steps))
        
        for inputs in train_loader:
            outputs = model(**inputs)
            loss = outputs.loss
            losses.append(loss.item())
                            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            progress_bar.update(1)
            
        loss = sum(losses) / len(losses)
        return loss
    
    
    def train(self, model:GPT2LMHeadModel, train_loader:DataLoader, val_loader:DataLoader, epochs:int) -> None:
        logger = SummaryWriter(log_dir=self.logs_dir)
        for e in range(1, epochs+1):
            print(f"\nEpoch: {e}")
            print("~"*160)
            
            train_loss = self._train_model(model, train_loader)
            val_loss = self._val_model(model, val_loader)
            
            self.checkpoint(val_loss, model)
            if self.lr_scheduler: self.lr_scheduler.step()
            
            logger.add_scalars("Losses", {"Train loss":train_loss, "Val loss":val_loss}, e)
            print(f"\nTrain loss: {train_loss:.4f} Val loss: {val_loss:.4f}")
        
        logger.close()