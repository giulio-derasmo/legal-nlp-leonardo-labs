#!/bin/bash
#PBS -S /bin/bash
#PBS -N "llama_training"
#PBS -q gpu
#PBS -l select=2:ncpus=48:ngpus=4:mpiprocs=4
#PBS -k oe
#PBS -j oe

########################################################################################################################
# Author: Giancarlo Paoletti (giancarlo.paoletti@leonardo.com) - feel free to ask me anything!
########################################################################################################################

module load cuda12.1
module load cudnn/8.9.7
module load miniconda3/py310_23.1.0-1
module load openmpi
module load proxy/proxy_20

conda init bash >/dev/null 2>&1
source ~/.bashrc >/dev/null 2>&1
conda activate giuder_ds >/dev/null 2>&1

########################################################################################################################
# Set variables for distributed training
########################################################################################################################
# Get current job id assigned from PBS
export JOBID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)

# Get the number of nodes and GPUs-per-node assigned from PBS
export NODE_LIST=($(sort "$PBS_NODEFILE" | uniq -c | xargs))
declare -a nodes=()
declare -a gpus=()
for item in "${NODE_LIST[@]}"; do if [[ $item =~ ^[0-9]+$ ]]; then gpus+=("$item"); else nodes+=("$item"); fi; done
export NODES=${#nodes[@]}

# Get master address and assign a free port by comparing two files and report unique lines only of first one.
# The first file is a sequence of ports in the specified range, sorted alphabetically.
# The second file checks for all used ports on $MASTER_ADDR and sorts alphabetically.
export MASTER_ADDR=${nodes[0]}
MASTER_PORT=($(ssh "$MASTER_ADDR" "comm -23  <(seq 20000 30000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1"))
export MASTER_PORT=${MASTER_PORT[-1]}

echo "############################################################"
echo "JOBID: $JOBID"
echo "NODES ASSIGNED: $NODES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
for ((i = 0; i < NODES; i++)); do
  GPUS=${gpus[$i]}
  NODE=${nodes[$i]}
  echo "GPUS ASSIGNED FOR NODE #$i ($NODE)--> $GPUS"
done
echo "############################################################"

# Set verbosity level for NCCL
if [ -z "$VERBOSITY_NCCL" ]; then VERBOSITY_NCCL="VERSION"; fi

# Set MPI environment variables
MPI_BASE_ENVS=""
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_MIN_NCHANNELS=16"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_NSOCKS_PERTHREAD=4"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_SOCKET_IFNAME=ib"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_SOCKET_NTHREADS=2"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_P2P_LEVEL=NVL"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NCCL_DEBUG=$VERBOSITY_NCCL"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x OMP_NUM_THREADS=1"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x MASTER_ADDR=$MASTER_ADDR"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x MASTER_PORT=$MASTER_PORT"
MPI_BASE_ENVS="${MPI_BASE_ENVS} -x NODES=$NODES"

# Set Accelerate environment variables
ACCELERATE_BASE_ENVS=""
# Optional arguments
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --quiet"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --dynamo_backend=no"
# Hardware and resource selection arguments
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --num_machines=1"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --main_process_ip=$MASTER_ADDR"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --main_process_port=$MASTER_PORT"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --mixed_precision=bf16"
# Training paradigm arguments#
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --multi_gpu"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --use_fsdp"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --use_deepspeed"
# FSDP arguments
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_offload_params=true"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_sharding_strategy=1" # FULL_SHARD
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_state_dict_type=FULL_STATE_DICT"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_forward_prefetch=false"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_use_orig_params=false"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --fsdp_sync_module_states=false"
# Deepspeed arguments
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --zero_stage=3"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --offload_optimizer_device=cpu"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --offload_param_device=cpu"
#ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --gradient_accumulation_steps=1"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --gradient_clipping=1.0"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --zero3_init_flag=true"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --zero3_save_16bit_model=true"
ACCELERATE_BASE_ENVS="${ACCELERATE_BASE_ENVS} --deepspeed_multinode_launcher=pdsh"

########################################################################################################################
# Execute program
########################################################################################################################
cd /davinci-1/home/gderasmo_ext/progetti/code

for ((i = 0; i < NODES; i++)); do
    export OMP_NUM_THREADS=1
    NODE_RANK=$i
    GPUS=${gpus[$i]}
    NODE=${nodes[$i]}

    # Update MPI variables
    MPI_RUN_ENVS="${MPI_BASE_ENVS} -x NODE_RANK=$NODE_RANK"

    # Update Accelerate variables
    ACCELERATE_ENVS="${ACCELERATE_BASE_ENVS} --num_processes=$GPUS"
    ACCELERATE_ENVS="${ACCELERATE_ENVS} --machine_rank=$NODE_RANK"
    ACCELERATE_ENVS="${ACCELERATE_ENVS} --rdzv_conf=rdzv_id=$JOBID,rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"

    # Gather executable for MPI
    MPI_EXEC="mpirun -q $MPI_RUN_ENVS --np 1 --bind-to none --map-by slot --mca pml ob1 --mca btl ^openib --H $NODE"

    # Gather executable for python scripts
    PYTHON_EXEC="accelerate launch $ACCELERATE_ENVS train_accelerate.py --quantization 8bit --max_length 2048 --version 2"

    # Gather overall executables
    EXEC="${MPI_EXEC} ${PYTHON_EXEC}"

    # Run it
    if ((i == 0)); then $EXEC & else ${EXEC} 2>/dev/null; fi
#    $EXEC &
    sleep 2s

done
wait

find /tmp -type d -name "*wandb*" -exec rm -rf {} \; 2>/dev/null
