#!/bin/bash 
# THIS SCRIPT IS A TEMPLATE!
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599

${PARTITION_OPTION}
${QOS_OPTION}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${OUTPUT_FILE}
#SBATCH --error=${ERROR_FILE}

${GIVEN_NODE}
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=${NUM_NODES}
####SBATCH --exclusive
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
${GPUS_PER_NODE_OPTION}
${CPUS_PER_NODE_OPTION}
${TIME_OPTION}
${MAIL_OPTION}

# ===== Loading the environment

${LOAD_ENV}

# TUNE_MAX_PENDING_TRIALS_PG=${MAX_PENDING_TRIALS}
# export TUNE_MAX_PENDING_TRIALS_PG
PYTHONHASHSEED=42
export PYTHONHASHSEED

# ===== Obtain the head IP address

REDIS_PASSWORD=$(uuidgen)
export REDIS_PASSWORD

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$HEAD_NODE_IP" == " " ]]; then
IFS=' ' read -ra ADDR <<<"$HEAD_NODE_IP"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  HEAD_NODE_IP=${ADDR[1]}
else
  HEAD_NODE_IP=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $HEAD_NODE_IP"
fi

# ===== Starting the Ray head node

# port=6379
port=6888
RAY_ADDRESS=auto
address_head=$HEAD_NODE_IP:$port
export HEAD_NODE_IP
export RAY_ADDRESS
echo "Address Head: $address_head"
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "REDIS_PASSWORD: $REDIS_PASSWORD"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$HEAD_NODE_IP" --port=$port --redis-password="$REDIS_PASSWORD" --block &
echo "SUCCESS"

# ===== Starting the Ray worker nodes

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Worker number = $worker_num"

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$address_head" --redis-password="$REDIS_PASSWORD" --block &
    echo "SUCCESS"

    sleep 5
done

# ===== Submitting the script
echo "Starting command: $COMMAND_PLACEHOLDER"
${COMMAND_PLACEHOLDER}
echo "SUCCESS"
