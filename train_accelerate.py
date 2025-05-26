# Author: Giancarlo Paoletti - LDO
#   Mail: giancarlo.paoletti@leonardo.com

import argparse
import builtins
import logging
import os

# --- monkey-patch torch.load to disable weights_only safety --------------------
import torch

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    logging.warning("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
import huggingface_hub
from trl import SFTConfig
from trl import SFTTrainer
from functools import partial
from transformers.trainer_utils import get_last_checkpoint
#torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])

import transformers
transformers.logging.set_verbosity_info()

m2hf = {
    # velvet models
    'velvet2b': 'Almawave/Velvet-2B',
    'velvet14b': 'Almawave/Velvet-14B',

    # llamantino models
    "llamantino_7b": 'swap-uniba/LLaMAntino-2-7b-hf-ITA',
    "llamantino_13b": 'swap-uniba/LLaMAntino-2-13b-hf-ITA',

    # qwen models
    'qwen_small': 'Qwen/Qwen2.5-0.5B',
    'qwen_1_5b': 'Qwen/Qwen2.5-1.5B',
    'qwen_2_5b': 'Qwen/Qwen2.5-3B',
    'qwen_7b': 'Qwen/Qwen2.5-7B',
    'qwen_14b': 'Qwen/Qwen2.5-14B',

    # llama models
    'llama3_8b': 'meta-llama/Meta-Llama-3-8B',
    'llama3_1_8b': 'meta-llama/Llama-3.1-8B',
    'llama3_2_1b': 'meta-llama/Llama-3.2-1B',
    'llama3_2_3b': 'meta-llama/Llama-3.2-3B',
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

accelerator = Accelerator()

original_print = builtins.print

def print_only_main(*args, **kwargs):
    if accelerator.is_main_process:
        builtins.print(*args, **kwargs)


builtins.print = print_only_main



def setup_logging():
    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.CRITICAL,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


#def hf_login():
#    huggingface_hub.login("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")

    
logger = setup_logging()
#hf_login()


def preprocess_function(example, tokenizer, max_length):
    tokenized =  tokenizer(example["text"], truncation=True, padding="max_length", max_length=max_length)
    #tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized
    
    
def train():
    parser = argparse.ArgumentParser(description='Train with Accelerate + DeepSpeed')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--train_filename', type=str, default='train.csv')
    parser.add_argument('--model_name', type=str, default="llama3_8b", help="short model name")
    parser.add_argument('--quantization', type=str, default="None", choices=["None", "4bit", "8bit"])
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--version', type=str, default="0", help="version of the train")
    parser.add_argument('--wandb_project', type=str, default="legalita")
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()
    
    args.max_length = 512
    base_model_id = m2hf[args.model_name]
    bnb_config = quant[args.quantization]
    
    set_seed(42)
    if accelerator.is_main_process:
      logger.info(f"Main path: {os.getcwd()}")
    
    if accelerator.is_main_process:
        logger.info(f"Running on: {accelerator.device}, distributed: {accelerator.distributed_type}")
        wandb.login(key='f37bda8ff53fbbb25723f6b1a35146b2ac6825fa')
        wandb.init(project=args.wandb_project, name=f"{args.model_name}-{args.version}")

    ########################################################
    # Load dataset
    #######################################################    
    #args.train_filename = 'train_normattiva.json'    
    dataset_path = os.path.join(args.data_path, args.train_filename)
    
    if accelerator.is_main_process:
        logger.info(f"{args.data_path}")
        logger.info(f"{os.path.join(args.data_path, args.train_filename)}")
      
    train_dataset = load_dataset('csv', data_files=dataset_path, split='train')
    #train_dataset = load_dataset('json', data_files=dataset_path)['train']
    #train_dataset = train_dataset.select(range(100_000))
    #train_dataset = train_dataset.select(range(100))
    
    if accelerator.is_main_process:
        logger.info(f"{train_dataset}")
    
    
    ########################################################
    # Load model & tokenizer
    ########################################################
    if accelerator.is_main_process:
        logger.info(f"Load model and tokenizer...")
        logger.info(f"{base_model_id} in {bnb_config} ...")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
       # cache_dir="/archive/group/ai/huggingface_shared_cache_labs/hub",
        cache_dir = '/davinci-1/home/gderasmo_ext/progetti/cache_dir'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
       # cache_dir="/archive/group/ai/huggingface_shared_cache_labs/hub",
        cache_dir = '/davinci-1/home/gderasmo_ext/progetti/cache_dir'
    )
    tokenizer.pad_token = tokenizer.eos_token


    if accelerator.is_main_process:
        logger.info(f"Tokenize the texts...")
    
    preprocess = partial(preprocess_function, tokenizer=tokenizer, max_length=args.max_length)
    train_dataset = train_dataset.map(preprocess, batched=True)

    if accelerator.is_main_process:
        logger.info(f"{train_dataset}")
        
    ########################################################
    # PEFT config (LoRA)
    ########################################################
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ########################################################
    # Training arguments
    ########################################################
    output_dir = os.path.join(args.log_path, f"legalita-{args.model_name}-{args.version}")
    
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    
    
    if accelerator.is_main_process:
        logger.info(f"{last_checkpoint}")
        
    training_arguments = SFTConfig(
        output_dir=output_dir,
        #label_names=["labels"],
        dataset_text_field="text",
        seed=3407,
        learning_rate=4.5e-4,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        weight_decay=0.01,
        optim="adamw_8bit",
        max_seq_length=args.max_length,
        packing=False,        
        save_strategy="steps", 
        save_steps=3500, 
        report_to=["wandb"],
        logging_steps=500,
        logging_dir=os.path.join(output_dir, "logs"), 
        push_to_hub=False,
    )
   
    ########################################################
    # Trainer
    ########################################################
    if accelerator.is_main_process:
        logger.info(f"Train...")
        
        
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        
    )
    trainer.train(resume_from_checkpoint=False)
  
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving model...")
        new_model = f"Romanai_512_{args.model_name}"
        trainer.model.save_pretrained(os.path.join(args.data_path, 'log/model', new_model), 
                                      save_embeddings=False)
        wandb.finish()


if __name__ == '__main__':
    train()