#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

date;hostname;id;pwd

which python

config_yaml='/private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/sweep_params.yaml'
echo 'config:' $config_yaml
python /private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/gen_config_yaml.py $config_yaml

train_file='/private/home/dejang/tools/loop_tool_env/loop_tool_service/models/cost_model/cost/cost_sweep_slurm.py'
echo 'train_file:' $train_file

project_name='wandb_slurm_torch'
echo 'project_name:' $project_name

echo 'running script'
python wandb_on_slurm.py $config_yaml $train_file $project_name