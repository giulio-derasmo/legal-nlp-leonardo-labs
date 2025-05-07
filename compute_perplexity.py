import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import numpy as np
import json
from huggingface_hub import login
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
# Log in using your token
login("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")

models2name = {
               'llama1b': "meta-llama/Llama-3.2-1B-Instruct",
               'llama8b': "meta-llama/Llama-3.1-8B-Instruct", 
               'llama70b': "meta-llama/Llama-3.1-70B-Instruct",
               'romanAI': 'giulioderasmo/RomanAI'
            } 


def load_model_and_tokenizer(model_name, load_in_4bit=False, load_in_8bit=True):
    
    READER_MODEL_NAME = models2name[model_name]


    if load_in_4bit:
        #  4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        #  8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else: 
        print('No quantization config provided, loading model in full precision')
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, 
                                                quantization_config=quantization_config,
                                                cache_dir='/home/fselab3/Documents/giuder/cache_dir',
                                                device_map="auto")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, 
                                            padding_side = 'right',)
    
    return model, tokenizer


def calculate_perplexity(model, tokenizer, test_dataset, prompt_length=20, max_seq_length=2048):
    model.eval()

    nlls = []
    # Process each sample individually instead of concatenating
    for i, sample in enumerate(tqdm(test_dataset["clean_text"])):
        # Tokenize sample and truncate to max_seq_length
        encoding = tokenizer(sample, return_tensors="pt", truncation=True, max_length=max_seq_length)
        input_ids = encoding.input_ids.to(model.device)

        # Skip samples that are too short
        if input_ids.size(1) <= prompt_length + 1:  # Need at least one token to predict
            continue

        # Split into prompt and target
        prompt_ids = input_ids[:, :prompt_length]
        target_ids = input_ids.clone()
        # Set tokens in the prompt section to -100 (ignore in loss calculation)
        target_ids[:, :prompt_length] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        # Accumulate loss * number of valid tokens
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())

    return ppl
l

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute perplexity for a dataset using a language model.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.", default='./data/test_ds.csv')
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--prompt_length", type=int, default=20, help="Length of the prompt.")
    args = parser.parse_args()

    args.load_in_4bit = False
    args.load_in_8bit = True

    print('load dataset')
    test_ds = pd.read_csv(args.dataset_name)
    _, test_ds = train_test_split(test_ds, test_size=1000, random_state=42, stratify=test_ds['label'])

    print('load model and tokenizer')
    model, tokenizer = load_model_and_tokenizer(args.model_name, load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit)
    max_seq_length = 1024

    print('compute perplexity')
    PPL = calculate_perplexity(model, tokenizer, test_ds.head(5000), prompt_length=20, max_seq_length=1024)

    print(f'PPL for the {models2name[args.model_name]} is {PPL:.3f}')

    with open('./perplexity.txt', 'a') as file:
        file.write(f"\n{args.model_name}\t{args.prompt_length}\t{max_seq_length}\t{PPL:.3f}") 