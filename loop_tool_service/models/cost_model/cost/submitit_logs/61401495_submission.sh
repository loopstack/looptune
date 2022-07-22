#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=/private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/submitit_logs/%j_0_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=submitit
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/submitit_logs/%j_0_log.out
#SBATCH --partition=learnlab
#SBATCH --signal=USR1@90
#SBATCH --time=20
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/submitit_logs/%j_%t_log.out --error /private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/submitit_logs/%j_%t_log.err /private/home/dejang/.conda/envs/compiler_gym/bin/python -u -m submitit.core._submit /private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/submitit_logs
