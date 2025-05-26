from transformers.trainer_utils import get_last_checkpoint
import os

last_checkpoint = None
log_path = './code/log'
model_name = 'llama3_8b'
version = 4
output_dir = os.path.join(log_path, f"legalita-{model_name}-{version}")

print(os.getcwd())
print("output_dir: ", output_dir)
print(os.listdir(output_dir))

last_checkpoint = get_last_checkpoint(output_dir)
if last_checkpoint:
    print(f"Resuming from checkpoint: {last_checkpoint}")
else:
    print("No checkpoint found, training from scratch.")


if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoint found, training from scratch.")
else:
    print("is not dir")