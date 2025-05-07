import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import wandb
from datetime import datetime
import huggingface_hub
import argparse



#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

m2hf = {## velvet models
        'velvet2b': 'Almawave/Velvet-2B',
        'velvet14b': 'Almawave/Velvet-14B',
        ## llamantino
        "llamantino_7b": 'swap-uniba/LLaMAntino-2-7b-hf-ITA',
        "llamantino_13b": 'swap-uniba/LLaMAntino-2-13b-hf-ITA',
        ## qwen models
        'qwen_small': 'Qwen/Qwen2.5-0.5B',
        'qwen_1_5b': 'Qwen/Qwen2.5-1.5B',
        'qwen_2_5b': 'Qwen/Qwen2.5-3B',
        'qwen_7b': 'Qwen/Qwen2.5-7B',
        'qwen_14b': 'Qwen/Qwen2.5-14B',
        ## llama models
        'llama3_8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3_1_8b': 'meta-llama/Llama-3.1-8B',
        'llama3_2_1b': 'meta-llama/Llama-3.2-1B',
        'llama3_2_3b': 'meta-llama/Llama-3.2-3B',

}

def wandb_init(args):
    wandb.login(key='f37bda8ff53fbbb25723f6b1a35146b2ac6825fa')
    wandb_project = f"legal-{args.model_name}"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

def hf_login():
    huggingface_hub.login("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")



def train():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_path', type=str, default='./data')   
    parser.add_argument('--train_filename', type=str, default='train.csv')  
    parser.add_argument('--log_path', type=str, default='./logs') 
    parser.add_argument('--model_name', type=str, required=True, help="short model name")
    parser.add_argument('--version', type=str, required=True, help="version of the train")
    args = parser.parse_args()


    ## wandb init
    wandb_init(args)
    hf_login()

    base_model_id = m2hf[args.model_name]
    print(args)
    
    ########################################################
    # Load dataset
    ########################################################
    train_dataset = load_dataset('csv', data_files=os.path.join(args.data_path, args.train_filename), split='train')
    #train_dataset = train_dataset.select(range(10000))
    #train_dataset = train_dataset.select(range(int(len(train_dataset)/2)))
    print(train_dataset)

    ########################################################
    # Load model
    ########################################################

    ## load model
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.bfloat16
    #)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit = True
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                 quantization_config=bnb_config, 
                                                 device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token


    max_length = 2048
    def preprocess_function(example):
        return tokenizer(example["clean_text"], truncation=True, max_length=max_length, padding="max_length")
    train_dataset = train_dataset.map(preprocess_function, batched=True)

    ########################################################
    # Set training parameters
    ########################################################
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    project = f"legalIta-{args.model_name}"
    output_dir = args.log_path + '/' + project


    training_arguments = SFTConfig(
        output_dir=output_dir,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate=4.5e-4, # Want a small lr for finetuning
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed = 3407,

        max_seq_length=max_length,
        
        #max_grad_norm=max_grad_norm= 0.3
        dataset_text_field="clean_text",
       
        packing=False,

        logging_steps=100,             
        logging_dir=args.log_path + '/log',      
        save_strategy="steps",                  
        save_steps=1500,                     

        report_to="wandb",             
        run_name=f"{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"    

    )

    ###W#####################################################
    # Train model
    ########################################################
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    save_model_name = f"RomanAI_base_{args.model_name}"
    trainer.model.save_pretrained(os.path.join('./models', save_model_name))

if __name__ == '__main__':
    train()