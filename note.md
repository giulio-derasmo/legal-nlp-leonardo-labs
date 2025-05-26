conda create --name giuder python=3.10
conda activate giuder
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install bitsandbytes accelerate datasets transformers peft trl wandb huggingface_hub


## deepspeed
conda create --name giuder_ds --clone giuder
downgrade pytorch  2.5 cuda12.1
pip install deepspeed

(sempre fare pip list per fare checking che è li nel env giusto)