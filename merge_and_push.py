from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os
import argparse
import huggingface_hub

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--hub_id", type=str)

    return parser.parse_args()
    

def main():
    #args = get_args()
    print(os.getcwd())

    huggingface_hub.login("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")
    #peft_model = './data/log/model/RomanAI_base_llama3_8b'
    #peft_model = './data/log/model/100k_legal_normattiva_dump_llama3_8b/'
    peft_model = './data/log/model/Romanai_512_llama3_8b/'
    print(os.listdir(peft_model))

    print(f"[0/5] Loading adpater config: {peft_model} ")
    config = PeftConfig.from_pretrained(peft_model)

    print(f"[1/5] Loading base model: {config.base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir = '/davinci-1/home/gderasmo_ext/progetti/cache_dir',
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    print(f"[2/5] Loading adapter: {peft_model}")
    model = PeftModel.from_pretrained(base_model, peft_model, device_map="auto")
    
    print("[3/5] Merge base model and adapter")
    model = model.merge_and_unload()
    
    #hub_id = 'giulioderasmo/RomanLLama'
    hub_id = "giulioderasmo/romanai_512_full"
    print(f"[4/5] Saving model and tokenizer in {hub_id}")
    #model.save_pretrained(f"{hub_id}")
    #tokenizer.save_pretrained(f"{hub_id}")

    print(f"[5/5] Uploading to Hugging Face Hub: {hub_id}")
    #hub_id_path = f'./{hub_id}'
    model.push_to_hub(f"{hub_id}")
    tokenizer.push_to_hub(f"{hub_id}")
    
    print("Merged model uploaded to Hugging Face Hub!")

if __name__ == "__main__" :
    main()