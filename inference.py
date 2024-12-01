import os
import torch
from typing import Tuple
from config import Config
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

load_dotenv()

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer() -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    model_dir = os.getenv("PRETRAINED_MODEL_DIR")
    tokenizer_dir = os.getenv("PRETRAINED_TOKENIZER_DIR")

    if not model_dir or not tokenizer_dir:
        raise ValueError("Environment variables 'PRETRAINED_MODEL_DIR' and 'PRETRAINED_TOKENIZER_DIR' must be set.")

    device = get_device()
    model = GPT2LMHeadModel.from_pretrained(model_dir, device_map=device)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir, device_map=device)
    return model, tokenizer


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, prompt: str, max_output_length: int = 300) -> str:
    config_dir = os.getenv("TOKENIZER_CONFIG_DIR")
    if not config_dir:
        raise ValueError("Environment variable 'TOKENIZER_CONFIG_DIR' must be set.")

    config = Config(config_dir).load()
    seq_length = config.get("seq_length", 512)

    input_text = f"<|startoftext|>Ingredients: {prompt.strip()}."
    inputs = tokenizer(input_text, truncation=True, max_length=seq_length, return_tensors="pt").to(device=model.device)

    outputs = model.generate(**inputs, max_length=max_output_length, do_sample=True, top_k=0, top_p=0.92, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(*outputs, skip_special_tokens=True).strip()
    return text


if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    while True:
        prompt = input("Enter ingredients: ")
        if prompt == "q": break
        generated_text = generate_text(model, tokenizer, prompt)
        print(generated_text + "\n")