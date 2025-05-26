module load miniconda3 
module load proxy
conda init
(chiudi e riapri terminale)
conda create env --name giuder python = 3.10
module load proxy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate peft trl datasets wandb bitsandbytes
chiudi e riapri

how to run the code: 
module load cuda12.1
module load cudnn/8.9.7
module load miniconda3/py310_23.1.0-1
module load openmpi
module load proxy/proxy_20
conda activate giuder
python code/train..... 


errors: 
- permission denied for archive/ai/hub
- [rank1]:[W508 15:19:59.517997423 ProcessGroupNCCL.cpp:4715] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
[rank2]:[W508 15:19:59.518099692 ProcessGroupNCCL.cpp:4715] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
[rank3]:[W508 15:19:59.520956767 ProcessGroupNCCL.cpp:4715] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.