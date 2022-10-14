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

import ray
from ray import tune
# from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.ppo import PPOTrainer

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


# Run this with: 
# python launcher/slurm_launch.py -e launcher/exp.yaml -n 1 -t 3:00   ### slurm_launch.py internaly calls rllib_torch.py
# python




parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--policy",  type=str, nargs='?', default='', help="Load policy network."
)
parser.add_argument(
    "--sweep",  type=int, nargs='?', const=1, default=0, help="Run with wandb sweeps"
)
parser.add_argument(
    "--slurm", 
    default=False, 
    action="store_true",
    help="Run on slurm"
)
parser.add_argument(
    "--iter", type=int, default=2, help="Number of iterations to train."
)
parser.add_argument("--size", type=int, nargs='?', default=1000000, help="Size of benchmarks to evaluate")

# parser.add_argument(
#     "--stop-timesteps", type=int, default=100, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=100, help="Reward at which we stop training."
# )
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Debuging",
)
parser.add_argument(
    "--local-mode",
    default=False,
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


default_config = {
    "log_level": "ERROR",
    "env": "compiler_gym", 
    "framework": 'torch',
    "model": {
        "custom_model": "my_model",
        "vf_share_layers": True,
        "fcnet_hiddens": [512] * 4,
        # "post_fcnet_hiddens":
        # "fcnet_activation": 
        # "post_fcnet_activation":
        # "no_final_linear":
        # "free_log_std":
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": torch.cuda.device_count(),
    # "num_workers": -1,  # parallelism
    "rollout_fragment_length": 100, 
    "train_batch_size": 7900, # train_batch_size == num_workers * rollout_fragment_length
    "num_sgd_iter": 50,
    # "evaluation_interval": 5, # num of training iter between evaluations
    # "evaluation_duration": 10, # num of episodes run per evaluation period
    "explore": True,
    "gamma": 0.9,
    "lr": 1e-6,
}



torch, nn = try_import_torch()
max_episode_steps = 20

def make_env():
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )

    env = TimeLimit(env, max_episode_steps=max_episode_steps) # <<<< Must be here
    return env

class RLlibAgent:
    def __init__(self, use_wandb=True) -> None:
        self.env = make_env()
        self.train_iter = max_episode_steps
        self.last_run_path=LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_wandb = use_wandb
        self.wandb_dict = {}


        ModelCatalog.register_custom_model(
            "my_model", my_net_rl.TorchCustomModel
        )


    def register_benchmark(self, train_benchmark):
        def make_training_env(*args): 
            del args
            return CycleOverBenchmarks(make_env(), train_benchmark)
        tune.register_env("compiler_gym", make_training_env)
        
    def load_datasets(self, datasets, data_size):
        train_benchmarks = []
        val_benchmarks = []
        for dataset in self.env.datasets.datasets():
            if dataset.name not in datasets:
                continue
            self.wandb_dict['dataset'] = dataset.name
            benchmarks = random.sample(list(dataset.benchmarks()), min(len(dataset), data_size))

            train_perc = 0.8
            train_size = int(np.ceil(train_perc * (len(benchmarks)-1) ))
            train_benchmarks.extend(benchmarks[:train_size]) 
            val_benchmarks.extend(benchmarks[train_size:])
            # train_benchmarks, val_benchmarks = torch.utils.data.random_split(benchmarks, [train_size, len(benchmarks) - train_size])

        print("Number of benchmarks for training:", len(train_benchmarks))
        print("Number of benchmarks for validation:", len(val_benchmarks))

        self.register_benchmark(train_benchmark=train_benchmarks)

        return train_benchmarks, val_benchmarks

    # from ray.tune import Callback
    # class MyCallback(Callback):
    #     def on_trial_result(self, iteration, trials, trial, result, **info):
    #         # breakpoint()
    #         print(f"Got result:")


    def train(self, algorithm, config, train_iter, sweep_count=1):
        """Training with RLlib agent.

        Args:
            algorithm (PPO.Trainer): PPO trainer.
            config (dict): config to run.
            train_iter (int): training iterations
            sweep_count (int, optional): number of sweeps. Defaults to 1.

        Returns:
            dict: [trial_id] = { "policy_path": policy_path, "config": config } after training
        """
        print(f'Before tune.run, stop = {train_iter}')
        models = {}
        callbacks = []
        self.train_iter = train_iter
        self.wandb_dict['algorithm'] = algorithm._name
        self.wandb_dict['actions'] = ",".join(self.env.action_space.names)

        if self.use_wandb:
            callbacks.append(WandbLoggerCallback(
                project="loop_tool_agent_split",
                # group=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                api_key_file=str(LOOP_TOOL_ROOT) + "/wandb_key.txt",
                log_config=False,
                )
            )

        analysis = tune.run(
            algorithm,
            metric="episode_reward_mean", # "final_performance",
            mode="max",
            reuse_actors=False,
            checkpoint_freq=10,
            checkpoint_at_end=True,
            config=config, 
            num_samples=max(1, sweep_count),
            stop={'training_iteration': train_iter},    
            callbacks=callbacks,
        )
        print("hhh2______________________")

        if os.path.exists(self.last_run_path):
            shutil.rmtree(self.last_run_path)

        os.makedirs(self.last_run_path)

        for trial in analysis.trials:
            config = trial.config
            config["explore"] = False
            agent = algorithm(
                env="compiler_gym",
                config=config
            )
            best_checkpoint = trial.checkpoint
            agent.restore(best_checkpoint.value)
            policy_model = agent.get_policy().model

            trial_dict = self.wandb_dict.copy()
            if 'fcnet_hiddens' in config['model']:
                trial_dict['layers_num'] = len(config['model']['fcnet_hiddens'])
                trial_dict['layers_width'] = config['model']['fcnet_hiddens'][0]

            models[trial.trial_id] = { 
                "policy_path": self.last_run_path/trial.trial_id/'policy_model.pt', 
                "config": trial_dict
            }
            
            os.makedirs(models[trial.trial_id]["policy_path"].parent)
            torch.save(policy_model,  models[trial.trial_id]["policy_path"])

            # weights_policy_path = LOOP_TOOL_ROOT/'loop_tool_service/models/weights/policy.pt'
            # if os.path.exists(weights_policy_path): os.remove(weights_policy_path)
            # os.symlink(self.last_run_path/'policy_model.pt', weights_policy_path)

        return models



#################################################################################
def wandb_update_df(wandb_dict, res_dict, prefix):
    wandb_dict[f'{prefix}final_performance'] = float(np.mean(res_dict['gflops']['greedy1_policy'] / res_dict['gflops']['greedy1_ln']))
    wandb_dict[f'{prefix}avg_search_base_speedup'] = float(np.mean(res_dict['gflops']['greedy1_ln'] / res_dict['gflops']['base']))
    wandb_dict[f'{prefix}avg_network_base_speedup'] = float(np.mean(res_dict['gflops']['greedy1_policy'] / res_dict['gflops']['base']))
    wandb_dict[f'{prefix}data_size'] = float(len(res_dict['gflops']))
    wandb_dict[f'{prefix}search_actions_num'] = float(np.mean(res_dict['actions']['greedy1_ln'].str.len()))
    wandb_dict[f'{prefix}network_actions_num'] = float(np.mean(res_dict['actions']['greedy1_policy'].str.len()))


def train(config, train_iter, sweep_count=1, policy_model_path=''):
    print(f'Train params: ', config, train_iter, policy_model_path)
    
    agent = RLlibAgent(use_wandb=True)

    train_benchmarks, val_benchmarks = agent.load_datasets(
        datasets=['benchmark://mm128_128_128-v0'],
        data_size=10000
    )
    breakpoint()
    models = agent.train(
        algorithm=PPOTrainer, 
        config=config, 
        train_iter=train_iter, 
        sweep_count=sweep_count
    )

    env = make_env()
    for trial_id, policy_model in models.items():
        evaluator = Evaluator(steps=2, cost_path="", policy_path=policy_model['policy_path'])

        breakpoint()
        
        train_dict = evaluator.evaluate(env, train_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.last_run_path/trial_id/"train")

        val_dict = evaluator.evaluate(env, val_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.last_run_path/trial_id/"validation")

        wandb_update_df(policy_model['config'], train_dict, prefix='train_')
        wandb_update_df(policy_model['config'], val_dict, prefix='')
        evaluator.send_to_wandb(path=agent.last_run_path/trial_id, wandb_run_id=trial_id, wandb_dict=policy_model['config'])



def update_default_config(sweep_config=None):
    for key, val in default_config.items():
        if key in sweep_config:
            if type(val) == dict:
                val.update(sweep_config[key])
            else:
                default_config[key] = sweep_config[key]

    return default_config
    


if __name__ == '__main__':
    # potential_policy = list(Path(last_run_path).parent.glob('**/policy_model.pt'))
    # policy_path = str(potential_policy[0]) if len(potential_policy) else ''

    args = parser.parse_args()


    # init_logging(level=logging.DEBUG)
    if ray.is_initialized(): ray.shutdown()

    print(f"Running with following CLI options: {args}")

    sweep_count = args.sweep
    
    if args.slurm:
        ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
        head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
        redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
        print('SLURM options: ', ray_address, head_node_ip, redis_password)
        ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)    
    else:
        ray.init(local_mode=args.local_mode, ignore_reinit_error=True)


    if 'num_workers' not in default_config: 
        default_config['num_workers'] = int(ray.cluster_resources()['CPU']) - 1

    if sweep_count and args.policy == '':
        hiddens_layers = [4]
        hiddens_width = [100, 500, 1000]
        sweep_config = {
            'lr': tune.uniform(1e-4, 1e-7),
            "gamma": tune.uniform(0.5, 0.99),
            'model': {
                "fcnet_hiddens": tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]),
            },

        }
        default_config = update_default_config(sweep_config)

    
    ############### Train ###############
    print(f'Train params: ', default_config, args.iter, args.policy)
    
    agent = RLlibAgent(use_wandb=True)

    train_benchmarks, val_benchmarks = agent.load_datasets(
        datasets=['benchmark://mm128_128_128-v0'],
        data_size=10000
    )

    models = agent.train(
        algorithm=PPOTrainer, 
        config=default_config, 
        train_iter=args.iter, 
        sweep_count=sweep_count
    )

    env = make_env()
    for trial_id, policy_model in models.items():
        evaluator = Evaluator(steps=2, cost_path="", policy_path=policy_model['policy_path'])
        
        df_train = evaluator.evaluate(env, train_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.last_run_path/trial_id/"train")

        df_val = evaluator.evaluate(env, val_benchmarks, { k:v for k, v in evaluator.searches.items() if 'cost' not in k })
        evaluator.save(path=agent.last_run_path/trial_id/"validation")
        breakpoint()
        wandb_update_df(policy_model['config'], df_train, prefix='train_')
        wandb_update_df(policy_model['config'], df_val, prefix='')
        evaluator.send_to_wandb(path=agent.last_run_path/trial_id, wandb_run_id=trial_id, wandb_dict=policy_model['config'])


    ray.shutdown()
    print("Return from train!")