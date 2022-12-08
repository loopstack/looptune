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

Reproduce results from wandb:
$ python rllib_agent.py --iter=0 --dataset=mm64_256_16_range  --wandb_url=dejang/loop_tool_agent_split/61e41_00000 --trainer=dqn.ApexTrainer  --eval_size=25 --eval_time=60
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
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register
from compiler_gym.wrappers import TimeLimit
import logging
from compiler_gym.util.logging import init_logging
from ray.tune.logger import Logger

import loop_tool_service
from loop_tool_service.models.evaluator import Evaluator


import torch
import importlib
import time

from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT
from os.path import exists
import wandb

import tempfile
import loop_tool as lt
# Run this with: 
# python rllib_agent.py --iter=2 --dataset=mm64_256_16_range
# python launcher/slurm_launch.py --app=rllib_agent.py --time=300:00 -nc=80 -ng=2 --iter=5000 --dataset=mm64_256_16_range --sweep  --steps=3
# python




parser = argparse.ArgumentParser()
parser.add_argument(
    '--trainer', type=str, default='ppo.PPOTrainer', help='The RLlib-registered trainer to use. Store config in rllib/config directory.'
)
parser.add_argument(
    "--wandb_url",  type=str, nargs='?', default='', help="Wandb uri to load policy network."
)
parser.add_argument(
    "--wandb_overwrite", default=False, action="store_true", help="Overwrite wandb results."
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
    "--steps", type=int, default=10, help="Number of actions to find."
)

parser.add_argument(
    "--dataset",  type=str, nargs='?', help="Dataset [mm128_128_128] to run must be defined in loop_tool_service.service_py.datasets.", required=True
)

parser.add_argument(
    '--network', choices=['TorchActionMaskModel', 'TorchBatchNormModel', 'TorchCustomModel'], default='TorchCustomModel', help='Deep network model.'
)

parser.add_argument(
    "--size", type=int, nargs='?', default=1000000, help="Size of benchmarks to train."
)

parser.add_argument(
    "--eval_size", type=int, default=100, help="Size of benchmarks to evaluate."
)

parser.add_argument(
    "--eval_time", type=int, default=10, help="Time to evaluate single benchmark."
)

parser.add_argument(
    "--stop_reward", type=float, default=1, help="Reward at which we stop training."
)

parser.add_argument(
    "--local-mode",
    default=False,
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


datasets_global = ['mm64_256_16_range']
max_episode_steps = 10



torch, nn = try_import_torch()
import ray.rllib.agents.trainer_template

def make_env():
    """Make the reinforcement learning environment for this experiment."""
    global datasets_global, max_episode_steps

    env = loop_tool_service.make(
        "loop_tool_env-v0",
        datasets=datasets_global,
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor_cached",# "flops_loop_nest_tensor",
    )

    env = TimeLimit(env, max_episode_steps=max_episode_steps) # <<<< Must be here
    return env


from ray.tune import Callback
class MyCallback(Callback):
    def __init__(self, agent):
        Callback.__init__(self)
        self.agent = agent

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if 'fcnet_hiddens' in trial.config['model']:
            self.agent.wandb_dict['layers_num'] = len(trial.config['model']['fcnet_hiddens'])
            self.agent.wandb_dict['layers_width'] = trial.config['model']['fcnet_hiddens'][0]

        self.agent.evaluator.send_to_wandb(wandb_run_id=trial.trial_id, wandb_dict=self.agent.wandb_dict)


class RLlibAgent:
    def __init__(self, trainer, dataset, size, eval_size, network, sweep_count, eval_time, wandb_key_path=str(LOOP_TOOL_ROOT) + "/wandb_key.txt") -> None:
        self.wandb_dict = {}

        global datasets_global, max_episode_steps
        # datasets_global = [ dataset ]
        self.max_episode_steps = max_episode_steps
        self.trainer_name = trainer
        self.algorithm, trainer = trainer.split('.') # expected: algorithm.trainer
        self.trainer = getattr(importlib.import_module(f'ray.rllib.agents.{self.algorithm}'), trainer)
        self.config = importlib.import_module(f"loop_tool_service.models.rllib.config.{self.algorithm}.{trainer}").get_config(sweep_count)
        self.dataset = dataset
        self.size = size
        self.eval_size = eval_size
        self.network = getattr(importlib.import_module(f"loop_tool_service.models.rllib.my_net_rl"), network)
        self.wandb_dict['network'] = network
        self.env = make_env()
        self.reward = list(self.env.reward.spaces.keys())[0]
        my_artifacts = Path(f'/private/home/dejang/tools/loop_tool_env/results/{time.strftime("%Y%m%d-%H%M%S")}') #Path(tempfile.mkdtemp()) # Dir to download and upload files. Has start, end subdirectories
        self.my_artifacts_start = my_artifacts/'start'
        self.my_artifacts_end = my_artifacts/'end'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wandb_key_path = wandb_key_path
        self.checkpoint_path = None
        self.policy_model = None
        self.train_benchmarks = []
        self.test_benchmarks = []
        self.checkpoint_start = None
        self.analysis = None
        self.evaluator = Evaluator(steps=max_episode_steps, reward=self.reward, timeout=eval_time)
        self.init()
    
    def init(self):
        # os.mkdir(self.my_artifacts_start)
        # os.mkdir(self.my_artifacts_end)
        self.my_artifacts_start.mkdir(parents=True)
        self.my_artifacts_end.mkdir(parents=True)
        
        ModelCatalog.register_custom_model(
            "my_model", self.network #my_net_rl.TorchBatchNormModel #if False else my_net_rl.TorchCustomModel
        )
        dataset =  self.env.datasets[f'benchmark://{self.dataset}-v0']
        benchmarks = [str(b) for b in dataset.benchmarks()]
        random.shuffle(benchmarks)
        benchmarks = benchmarks[:self.size]
        self.wandb_dict['dataset'] = dataset.name
        self.wandb_dict['max_episode_steps'] = self.max_episode_steps
        
        train_perc = 0.8
        train_size = int(np.ceil(train_perc * (len(benchmarks)-1) ))
        
        self.train_benchmarks = benchmarks[:train_size]
        self.test_benchmarks = benchmarks[train_size:]
        self.wandb_dict['train_benchmarks'] = self.train_benchmarks
        self.wandb_dict['test_benchmarks'] = self.test_benchmarks
        self.wandb_dict['eval_train_benchmarks'] = self.train_benchmarks[:self.eval_size]
        self.wandb_dict['eval_test_benchmarks'] = self.test_benchmarks[:self.eval_size]

        self.wandb_dict['train_size'] = len(self.train_benchmarks)
        self.wandb_dict['test_size'] = len(self.test_benchmarks)
        self.wandb_dict['reward'] = self.reward

        self.wandb_dict['max_loops'] = lt.LoopTreeAgent.max_loops()
        self.wandb_dict['num_loop_features'] = lt.LoopTreeAgent.num_loop_features()
        self.wandb_dict['trainer'] = self.trainer.__name__
        self.wandb_dict['actions'] = ",".join(self.env.action_space.names)

        print("Number of benchmarks for training:", len(self.train_benchmarks))
        print("Number of benchmarks for test:", len(self.test_benchmarks))
                    

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
            self.checkpoint_path = f"{self.my_artifacts_start}/{self.checkpoint_start['checkpoint']}"
            self.config['model']['fcnet_hiddens'] = [self.checkpoint_start['layers_width']] * self.checkpoint_start['layers_num']

            for f in wandb_run.files(): 
                if f.name.startswith('checkpoint'):
                    f.download(root=self.my_artifacts_start, replace=True)

            self.train_benchmarks = self.checkpoint_start['train_benchmarks']
            self.test_benchmarks = self.checkpoint_start['test_benchmarks']
            self.wandb_dict['eval_train_benchmarks'] = self.checkpoint_start['eval_train_benchmarks'][:self.eval_size] if self.eval_size < len(self.checkpoint_start['eval_train_benchmarks']) else self.checkpoint_start['train_benchmarks'][:self.eval_size]
            self.wandb_dict['eval_test_benchmarks'] = self.checkpoint_start['eval_test_benchmarks'][:self.eval_size] if self.eval_size < len(self.checkpoint_start['eval_test_benchmarks']) else self.checkpoint_start['eval_test_benchmarks'][:self.eval_size]


        except:
            print('Wandb had problem with loading the model!')
            return

            


            
    def train(self, train_iter, stop_reward=1, sweep_count=1):
        """Training with RLlib agent.

        Args:
            config (dict): config to run.
            train_iter (int): training iterations
            sweep_count (int, optional): number of sweeps. Defaults to 1.

        Returns:
            dict: [trial_id] = { "policy_path": policy_path, "config": config } after training
        """
        if train_iter <= 0:
            return

        if self.checkpoint_start != None:
            train_iter += self.checkpoint_start['training_iteration']

        if train_iter:
            self.analysis = tune.run(
                self.trainer,
                config=self.config, 
                restore=self.checkpoint_path,
                metric="episode_reward_mean", # "final_performance",
                mode="max",
                reuse_actors=False,
                checkpoint_freq=10,
                checkpoint_at_end=True,
                num_samples=max(1, sweep_count),
                stop={'training_iteration': train_iter, 'episode_reward_mean': stop_reward},   
                callbacks=[
                    MyCallback(self),
                    WandbLoggerCallback(
                        project="loop_tool_agent_split",
                        api_key_file=self.wandb_key_path,
                        log_config=False,
                    )
                ],
            )
            print("hhh2______________________")


    def evaluate(self, searches, wandb_overwrite=False):
        if self.analysis:
            config_checkpoint_pairs = [ [trial.trial_id, trial.config, str(trial.checkpoint.value)] for trial in self.analysis.trials ]
        elif self.checkpoint_path != None:
            config_checkpoint_pairs = [[ self.wandb_dict['wandb_start'].split('/')[-1], self.config, self.checkpoint_path]]
        else:
            print('RLlibAgent: No analysis found. Run train method first.')
            return


        searches_cmd = { k:v for k, v in self.evaluator.searches.items() if k in searches} 

        for trial_id, config, checkpoint_path_str in config_checkpoint_pairs:
            if checkpoint_path_str == None:
                continue

            checkpoint_path = Path(checkpoint_path_str) # .value -> .dir_or_data for ray 2.1
            
            config["explore"] = False
            agent = self.trainer(
                env="compiler_gym",
                config=config
            )

            agent.restore(str(checkpoint_path))

            if 'fcnet_hiddens' in config['model']:
                self.wandb_dict['layers_num'] = len(config['model']['fcnet_hiddens'])
                self.wandb_dict['layers_width'] = config['model']['fcnet_hiddens'][0]

            self.wandb_dict['checkpoint'] = os.path.relpath(checkpoint_path, checkpoint_path.parent.parent)

            # Save policy and checkpoint for wandb
            policy_path = self.my_artifacts_end/trial_id/'policy_model.pt'
            os.makedirs(policy_path.parent)
            my_artifacts_checkpoint_dir = self.my_artifacts_end/trial_id/checkpoint_path.parent.name
            shutil.copytree(checkpoint_path.parent, my_artifacts_checkpoint_dir)
            with open(my_artifacts_checkpoint_dir/'config.json', "w") as f: json.dump(config, f)

            if wandb_overwrite:
                self.evaluator.send_to_wandb(wandb_run_id=trial_id, wandb_dict=self.wandb_dict, path=self.my_artifacts_end/trial_id, overwrite=wandb_overwrite)

            self.evaluator.set_policy_agent(agent)

            df_train = self.evaluator.evaluate(self.env, self.wandb_dict['eval_train_benchmarks'], searches_cmd)
            self.evaluator.save(path=self.my_artifacts_end/trial_id/"train")
            df_val = self.evaluator.evaluate(self.env, self.wandb_dict['eval_test_benchmarks'], searches_cmd)
            self.evaluator.save(path=self.my_artifacts_end/trial_id/"test")
        
            self.wandb_update_df(df_train, prefix='train_')
            self.wandb_update_df(df_val, prefix='test_')
            if wandb_overwrite:
                self.evaluator.send_to_wandb(wandb_run_id=trial_id, wandb_dict=self.wandb_dict, path=self.my_artifacts_end/trial_id, overwrite=wandb_overwrite)
            print(f'Saved at: {self.my_artifacts_end}')


    def wandb_update_df(self, res_dict, prefix):
        self.wandb_dict[f'{prefix}final_performance'] = float(np.mean(res_dict['gflops']['loop_tune_ln'].str[-1] / res_dict['gflops']['greedy1_ln'].str[-1]))
        self.wandb_dict[f'{prefix}avg_search_base_speedup'] = float(np.mean(res_dict['gflops']['greedy1_ln'].str[-1] / res_dict['gflops']['base'].str[-1]))
        self.wandb_dict[f'{prefix}avg_network_base_speedup'] = float(np.mean(res_dict['gflops']['loop_tune_ln'].str[-1] / res_dict['gflops']['base'].str[-1]))
        self.wandb_dict[f'{prefix}search_actions_num'] = float(np.mean(res_dict['actions']['greedy1_ln'].str.len()))
        self.wandb_dict[f'{prefix}network_actions_num'] = float(np.mean(res_dict['actions']['loop_tune_ln'].str.len()))

#################################################################################



if __name__ == '__main__':
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")
    
    # global max_episode_steps
    # max_episode_steps = args.steps


    # init_logging(level=logging.DEBUG)
    if ray.is_initialized(): ray.shutdown()
    global max_num_steps
    
    if args.slurm:
        ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
        head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
        redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
        print('SLURM options: ', ray_address, head_node_ip, redis_password)
        ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)    
    else:
        ray.init(local_mode=args.local_mode, ignore_reinit_error=True)


    agent = RLlibAgent(
        trainer=args.trainer, 
        dataset=args.dataset, 
        # steps=args.steps,
        size=args.size, 
        eval_size=args.eval_size,
        network=args.network, 
        sweep_count=args.sweep, 
        eval_time=args.eval_time
    )

    if args.wandb_url:
        agent.load_model(args.wandb_url)

    agent.train(
        train_iter=args.iter, 
        stop_reward=args.stop_reward,
        sweep_count=args.sweep,
    )
    agent.evaluate(
        searches=['greedy1_ln', 'greedy2_ln', 'beam2dfs_ln', 'beam4dfs_ln','beam2bfs_ln', 'beam4bfs_ln', 'random_ln', 'loop_tune_ln'],    
        wandb_overwrite=args.wandb_overwrite,
    )

    ray.shutdown()
    print("Return from train!")