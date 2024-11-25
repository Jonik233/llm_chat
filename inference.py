import os
import torch
import argparse
from typing import Tuple
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load() -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    load_dotenv()
    model_dir = os.getenv("PRETRAINED_MODEL_DIR")
    tokenizer_dir = os.getenv("PRETRAINED_TOKENIZER_DIR")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(model_dir, device_map=device)
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir, device_map=device)
    return model, tokenizer

def inference(model:GPT2LMHeadModel, tokenizer:GPT2TokenizerFast, prompt:str, output_length:int=300) -> str:
    input_sentence = f"<|startoftext|>Ingredients: {prompt.strip()}."
    inputs = tokenizer(input_sentence, return_tensors="pt").to(device=model.device)
    outputs = model.generate(**inputs, max_length=output_length, do_sample=True, top_k=0, top_p=0.92, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(*outputs, skip_special_tokens=True)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference function")
    parser.add_argument("prompt", type=str, help="Prompt for generation")
    args = parser.parse_args()
    
    prompt = args.prompt
    model, tokenizer = load()
    text = inference(model, tokenizer, prompt)
    print(text)