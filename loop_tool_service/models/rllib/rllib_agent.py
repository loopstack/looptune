"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import ast
from distutils.command.config import config
from math import ceil, floor
import gym
from itertools import islice
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import shutil
import json
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
from copy import deepcopy
import yaml

import ray
from ray import tune
# from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register
from compiler_gym.wrappers import TimeLimit
import logging
from compiler_gym.util.logging import init_logging
from ray.tune.logger import Logger

import loop_tool_service
from loop_tool_service import paths
from loop_tool_service.models.evaluator import Evaluator

from loop_tool_service.service_py.datasets import mm128_128_128

from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
import loop_tool_service.models.rllib.my_net_rl as my_net_rl

import torch
from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT
from os.path import exists
import wandb

import tempfile

# Run this with: 
# python launcher/slurm_launch.py -e launcher/exp.yaml -n 1 -t 3:00   ### slurm_launch.py internaly calls rllib_torch.py
# python




parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--wandb_url",  type=str, nargs='?', default='', help="Wandb uri to load policy network."
)
parser.add_argument(
    "--sweep",  type=int, nargs='?', const=1, default=0, help="Run with wandb sweeps."
)
parser.add_argument(
    "--slurm", 
    default=False, 
    action="store_true",
    help="Run on slurm."
)
parser.add_argument(
    "--iter", type=int, default=2, help="Number of iterations to train."
)

parser.add_argument(
    "--dataset",  type=str, nargs='?', help="Dataset [mm128_128_128] to run must be defined in loop_tool_service.service_py.datasets.", required=True
)

parser.add_argument(
    "--size", type=int, nargs='?', default=1000000, help="Size of benchmarks to evaluate."
)

# parser.add_argument(
#     "--stop-timesteps", type=int, default=100, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=100, help="Reward at which we stop training."
# )

parser.add_argument(
    "--local-mode",
    default=False,
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


default_config = {
    "log_level": "ERROR",
    "env": "compiler_gym", 
    "observation_space": "loops_tensor",
    "framework": 'torch',
    "model": {
        "custom_model": "my_model",
        "custom_model_config": {"action_mask": [1, 0, 0, 0]},
        "vf_share_layers": True,
        "fcnet_hiddens": [1000] * 4,
        # "post_fcnet_hiddens":
        # "fcnet_activation": 
        # "post_fcnet_activation":
        # "no_final_linear":
        # "free_log_std":
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), #torch.cuda.device_count(),
    # "num_workers": -1,  # parallelism
    "rollout_fragment_length": 10, 
    "train_batch_size": 790, # train_batch_size == num_workers * rollout_fragment_length
    "num_sgd_iter": 30,
    # "evaluation_interval": 5, # num of training iter between evaluations
    # "evaluation_duration": 10, # num of episodes run per evaluation period
    "explore": True,
    "gamma": 0.7689098370203395,
    "lr": 3.847293324197388e-7,
}



torch, nn = try_import_torch()
max_episode_steps = 4

datasets_global = None

def make_env():
    """Make the reinforcement learning environment for this experiment."""
    # if dataset == []:
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        datasets=datasets_global,
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )

    env = TimeLimit(env, max_episode_steps=max_episode_steps) # <<<< Must be here
    return env

class RLlibAgent:
    def __init__(self, algorithm, dataset, wandb_key_path=str(LOOP_TOOL_ROOT) + "/wandb_key.txt") -> None:
        global datasets_global
        self.algorithm = algorithm
        datasets_global = [ dataset ]
        self.dataset = dataset
        self.env = make_env()
        self.train_iter = max_episode_steps
        self.my_artifacts=LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wandb_key_path = wandb_key_path
        self.wandb_dict = {}
        self.policy_model = None
        self.train_benchmarks = []
        self.validation_benchmarks = []
        self.checkpoint_start = None
        self.tempdir = tempfile.mkdtemp()
        self.init()
    
    def init(self):
        ModelCatalog.register_custom_model(
            "my_model", my_net_rl.TorchActionMaskModel #if False else my_net_rl.TorchCustomModel
        )
        dataset =  self.env.datasets[f'benchmark://{self.dataset}-v0']
        benchmarks = list(dataset.benchmarks())
        self.wandb_dict['dataset'] = dataset.name
        
        train_perc = 0.8
        train_size = int(np.ceil(train_perc * (len(benchmarks)-1) ))
        random.shuffle(benchmarks)
        self.train_benchmarks = sorted(benchmarks[:train_size])
        self.validation_benchmarks = sorted(benchmarks[train_size:])
        self.wandb_dict['train_size'] = len(self.train_benchmarks)
        self.wandb_dict['test_size'] = len(self.validation_benchmarks)

        print("Number of benchmarks for training:", len(self.train_benchmarks))
        print("Number of benchmarks for validation:", len(self.validation_benchmarks))

        def make_training_env(*args): 
            del args
            return CycleOverBenchmarks(make_env(), benchmarks[:train_size])
        tune.register_env("compiler_gym", make_training_env)

    def make_env(self):
        return make_env()


    def load_model(self, wandb_url):
        try:
            api = wandb.Api()
            wandb_run = api.run(wandb_url)
            self.wandb_dict['wandb_start'] = wandb_url
            self.checkpoint_start = wandb_run.summary
            

            for f in wandb_run.files(): 
                if f.name.startswith('checkpoint'):
                    f.download(root=self.tempdir, replace=True)

            # wandb.restore('config.yaml', run_path=policy_path)
            # policy_model = torch.load('policy_model.pt')
            # with open('config.yaml', 'r') as f: config = yaml.load(f, Loader=yaml.BaseLoader)

        except:
            print('Policy not found')


    def train(self, config, train_iter, sweep_count=1):
        """Training with RLlib agent.

        Args:
            config (dict): config to run.
            train_iter (int): training iterations
            sweep_count (int, optional): number of sweeps. Defaults to 1.

        Returns:
            dict: [trial_id] = { "policy_path": policy_path, "config": config } after training
        """
        print(f'Before tune.run, stop = {train_iter}')
        models = {}
        self.train_iter = train_iter
        self.wandb_dict['algorithm'] = self.algorithm.__name__
        self.wandb_dict['actions'] = ",".join(self.env.action_space.names)

        checkpoint_path = None
        if self.checkpoint_start != None:
            train_iter += self.checkpoint_start['training_iteration']
            checkpoint_path = f"{self.tempdir}/{self.checkpoint_start['checkpoint']}"
            config['model']['fcnet_hiddens'] = [self.checkpoint_start['layers_width']] * self.checkpoint_start['layers_num']

        analysis = tune.run(
            self.algorithm,
            restore=checkpoint_path,
            metric="episode_reward_mean", # "final_performance",
            mode="max",
            reuse_actors=False,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            config=config, 
            num_samples=max(1, sweep_count),
            stop={'training_iteration': train_iter},    
            callbacks=[
                WandbLoggerCallback(
                    project="loop_tool_agent_split",
                    api_key_file=self.wandb_key_path,
                    log_config=False,
                )
            ],
        )
        print("hhh2______________________")
        breakpoint()

        if os.path.exists(self.my_artifacts):
            shutil.rmtree(self.my_artifacts)

        os.makedirs(self.my_artifacts)

        for trial in analysis.trials:
            config = trial.config
            config["explore"] = False
            agent = self.algorithm(
                env="compiler_gym",
                config=config
            )

            checkpoint_path = Path(trial.checkpoint.dir_or_data)
            agent.restore(str(checkpoint_path))
            policy_model = agent.get_policy().model

            trial_dict = self.wandb_dict.copy()
            if 'fcnet_hiddens' in config['model']:
                trial_dict['layers_num'] = len(config['model']['fcnet_hiddens'])
                trial_dict['layers_width'] = config['model']['fcnet_hiddens'][0]

            trial_dict['checkpoint'] = os.path.relpath(checkpoint_path, checkpoint_path.parent.parent)
            
            models[trial.trial_id] = { 
                "policy_path": self.my_artifacts/trial.trial_id/'policy_model.pt', 
                "config": trial_dict
            }          

            # Save policy and checkpoint for wandb
            os.makedirs(models[trial.trial_id]["policy_path"].parent)
            torch.save(policy_model,  models[trial.trial_id]["policy_path"])
            my_artifacts_checkpoint_dir = self.my_artifacts/trial.trial_id/checkpoint_path.parent.name
            shutil.copytree(checkpoint_path.parent, my_artifacts_checkpoint_dir)
            with open(my_artifacts_checkpoint_dir/'config.json', "w") as f: json.dump(trial.config, f)


        return models



#################################################################################
def wandb_update_df(wandb_dict, res_dict, prefix):
    wandb_dict[f'{prefix}final_performance'] = float(np.mean(res_dict['gflops']['greedy1_policy'] / res_dict['gflops']['greedy1_ln']))
    wandb_dict[f'{prefix}avg_search_base_speedup'] = float(np.mean(res_dict['gflops']['greedy1_ln'] / res_dict['gflops']['base']))
    wandb_dict[f'{prefix}avg_network_base_speedup'] = float(np.mean(res_dict['gflops']['greedy1_policy'] / res_dict['gflops']['base']))
    wandb_dict[f'{prefix}search_actions_num'] = float(np.mean(res_dict['actions']['greedy1_ln'].str.len()))
    wandb_dict[f'{prefix}network_actions_num'] = float(np.mean(res_dict['actions']['greedy1_policy'].str.len()))


def update_default_config(sweep_config=None):
    for key, val in default_config.items():
        if key in sweep_config:
            if type(val) == dict:
                val.update(sweep_config[key])
            else:
                default_config[key] = sweep_config[key]

    return default_config
    

def train(config, dataset, iter, wandb_url, sweep_count):
    ############### Train ###############
    print(f'Train params: ', config, iter, wandb_url)

    agent = RLlibAgent(algorithm=PPOTrainer, dataset=dataset)

    if wandb_url:
        agent.load_model(wandb_url)

    models = agent.train(
        config=config, 
        train_iter=iter, 
        sweep_count=sweep_count
    )

    env = agent.make_env()
    for trial_id, policy_model in models.items():
        evaluator = Evaluator(steps=max_episode_steps, cost_path="", policy_path=policy_model['policy_path'])
        
        df_train = evaluator.evaluate(env, agent.train_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.my_artifacts/trial_id/"train")

        df_val = evaluator.evaluate(env, agent.validation_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.my_artifacts/trial_id/"validation")
        wandb_update_df(policy_model['config'], df_train, prefix='train_')
        wandb_update_df(policy_model['config'], df_val, prefix='test_')
        evaluator.send_to_wandb(path=agent.my_artifacts/trial_id, wandb_run_id=trial_id, wandb_dict=policy_model['config'])




if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    # init_logging(level=logging.DEBUG)
    if ray.is_initialized(): ray.shutdown()

    
    if args.slurm:
        ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
        head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
        redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
        print('SLURM options: ', ray_address, head_node_ip, redis_password)
        ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)    
    else:
        ray.init(local_mode=args.local_mode, ignore_reinit_error=True)


    print(f"Num of CPUS = {int(ray.cluster_resources()['CPU'])}")
    print(f'Num of GPUS = {torch.cuda.device_count()}, ray = {ray.get_gpu_ids()}')


    if 'num_workers' not in default_config: 
        default_config['num_workers'] = int(ray.cluster_resources()['CPU']) - 1

    if args.sweep:
        hiddens_layers = [8, 16]
        hiddens_width = [100, 500, 1000]
        sweep_config = {
            'lr': tune.uniform(1e-6, 1e-8),
            "gamma": tune.uniform(0.7, 0.99),
            'model': {
                "fcnet_hiddens": tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]),
            },
        }
        default_config = update_default_config(sweep_config)

    train(config=default_config, dataset=args.dataset, iter=args.iter, wandb_url=args.wandb_url, sweep_count=args.sweep)


    ray.shutdown()
    print("Return from train!")