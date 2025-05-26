import transformers
from peft import PeftModel, PeftConfig
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
import os
# Log in using your token
login("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")

m2hf = {'base_model': {  
                         'llama8b': "meta-llama/Meta-Llama-3-8B", 
                         'romanLLama': 'giulioderasmo/RomanLLama',
                         '100klegal': 'giulioderasmo/test_100k_legalai',
                         'romanLLama512': 'giulioderasmo/RomanLLama',
                       },
        'adapter': { 'romanLLama_adapter': 'giulioderasmo/RomanLLama-adapter', }
     } 

quant = {
    "None": None,
    "4bit": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    "8bit": BitsAndBytesConfig(
        load_in_8bit=True
    ),
}

def load_model_and_tokenizer(args):

    if not args.is_adapter: 
        base_model_id = m2hf['base_model'][args.model_name]
        adapter_id = None
    else: 
        adapter_id = m2hf['adapter'][args.model_name]
        config = PeftConfig.from_pretrained(adapter_id)
        base_model_id = config.base_model_name_or_path
         
    bnb_config = quant[args.quantization]
    model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                quantization_config=bnb_config,
                                                cache_dir = '/davinci-1/home/gderasmo_ext/progetti/cache_dir',
                                                device_map="auto")

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, 
                                              padding_side = 'right',)
    
    if args.is_adapter:
        print('load adapter')
        model = PeftModel.from_pretrained(model, adapter_id, device_map="auto")
    
    model.eval()
    return model, tokenizer


def ppl_model(model, tokenizer, dataset):
  nlls= []
  max_length = 1024
  stride = 512
  for s in tqdm(range(len(dataset['text']))):
      encodings = tokenizer(dataset['text'][s], return_tensors="pt")
      seq_len = encodings.input_ids.size(1)
      prev_end_loc = 0
      for begin_loc in range(0, seq_len, stride):
          end_loc = min(begin_loc + max_length, seq_len)
          trg_len = end_loc - prev_end_loc
          input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
          target_ids = input_ids.clone()
          target_ids[:, :-trg_len] = -100
          with torch.no_grad():
              outputs = model(input_ids, labels=target_ids)
              neg_log_likelihood = outputs.loss
          nlls.append(neg_log_likelihood)
          prev_end_loc = end_loc
          if end_loc == seq_len:
              break
  ppl = torch.exp(torch.stack(nlls).mean())
  return ppl


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute perplexity for a dataset using a language model.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to use.", default='./data/test.csv')
    #parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    #parser.add_argument("--is_adapter", type=int, default=0, choices=[0, 1])
    parser.add_argument('--quantization', type=str, default="None", choices=["None", "4bit", "8bit"])
    args = parser.parse_args()
    
    args.prompt_length = 20
    args.max_seq_length = 512
    
    for model_name, is_adapter in zip(['romanLLama512', '100klegal', 'llama8b', 'romanLLama_adapter', 'romanLLama', ], [0, 0, 0, 1, 0]):
        args.model_name = model_name
        args.is_adapter = is_adapter
        print(args)
        
        print('load dataset')
        test_ds = pd.read_csv(args.dataset_name)
        print('test_ds pre split', test_ds)
        #_, test_ds = train_test_split(test_ds, test_size=1500, random_state=42, stratify=test_ds['label'])
        
        print('test_ds', test_ds)
        
        #args.train_filename = 'train_normattiva.json'    
        #dataset_path = os.path.join('./data', args.train_filename)
        #train_dataset = load_dataset('json', data_files=dataset_path)['train']
        #test_ds = train_dataset.select(range(100_000, 101_000))
    
        print('load model and tokenizer')
        model, tokenizer = load_model_and_tokenizer(args)
    
        print('compute perplexity')
        PPL = ppl_model(model, tokenizer, test_ds)
    
        print(f'PPL for the {args.model_name} is {PPL:.3f}')
        
        with open('./perplexity.txt', 'a') as file:
            file.write(f"\n{args.model_name}\t{PPL:.3f}") 