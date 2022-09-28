'''
In this experiment we aim to compare: 
    - train_time
    - compile_time
    - execution_time

for the following search algorithms:
    - LoopNest
    - BruteForce(10 actions)
    - Greedy(0 patience, 10 walks, 10 actions) 
    - Greedy(1 patience, 1 walks, 10 actions)
    - Policy(10 actions)
    - PolicyBeam(10 actions) 
'''

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

from loop_tool_service.service_py.datasets import single_mmo_dataset, full_mmo_dataset, single_mm_dataset, loop_tool_dataset, loop_tool_test_dataset

from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
import loop_tool_service.models.rllib.my_net_rl as my_net_rl


import wandb
from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT


# Run this with: 
# python launcher/slurm_launch.py -e launcher/exp.yaml -n 1 -t 3:00   ### slurm_launch.py internaly calls rllib_torch.py
# python

torch, nn = try_import_torch()



last_run_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
device = 'cuda' if torch.cuda.is_available() else 'cpu'




parser = argparse.ArgumentParser()
parser.add_argument(
    "--policy",  type=str, nargs='?', const=str(list(Path(last_run_path).glob('**/policy_model.pt'))[0]) , default='', help="Load policy network."
)

parser.add_argument(
    "--steps", type=int, default=20, help="Length of sequence of actions to evaluate"
)
parser.add_argument("--size", type=int, nargs='?', default=1000000, help="Size of benchmarks to evaluate")


args = parser.parse_args()




def register_env():
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.NormRewardTensor(),
                ],
            "datasets": [
                single_mmo_dataset.Dataset(),
                # full_mmo_dataset.Dataset(),
            ],
        },
    )


def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )
    # env = compiler_gym.make("loop_tool_env-v0")

    env = TimeLimit(env, max_episode_steps=args.steps) # <<<< Must be here
    return env


def load_datasets(env=None):

    with make_env() as env:
        train_benchmarks = []
        val_benchmarks = []
        for dataset in env.datasets.datasets():
            wandb_log['dataset'] = dataset.name
            benchmarks = random.sample(list(dataset.benchmarks()), min(len(dataset), args.size))

            train_perc = 0.8
            train_size = int(np.ceil(train_perc * (len(benchmarks)-1) ))
            train_benchmarks.extend(benchmarks[:train_size]) 
            val_benchmarks.extend(benchmarks[train_size:])
            # train_benchmarks, val_benchmarks = torch.utils.data.random_split(benchmarks, [train_size, len(benchmarks) - train_size])

        print("Number of benchmarks for training:", len(train_benchmarks))
        print("Number of benchmarks for validation:", len(val_benchmarks))    
        return train_benchmarks, val_benchmarks

from ray.tune import Callback
class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        # breakpoint()
        print(f"Got result:")


def train_agent(config, stop_criteria, sweep_count=1):
    print(f'Before tune.run, stop = {stop_criteria}')
    models = {}
    analysis = tune.run(
        PPOTrainer,
        metric="episode_reward_mean", # "final_performance",
        mode="max",
        reuse_actors=False,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config=config, 
        num_samples=max(1, sweep_count),
        stop=stop_criteria,    
        callbacks=[ 
            # MyCallback(),
            WandbLoggerCallback(
                project="loop_tool_agent_split",
                # group=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                api_key_file=str(LOOP_TOOL_ROOT) + "/wandb_key.txt",
                log_config=False,
                )
        ]
    )
    print("hhh2______________________")

    if os.path.exists(last_run_path):
        shutil.rmtree(last_run_path)

    os.makedirs(last_run_path)

    for trial in analysis.trials:
        os.makedirs(last_run_path/trial.trial_id)
        config = trial.config
        config["explore"] = False
        agent = PPOTrainer(
            env="compiler_gym",
            config=config
        )
        best_checkpoint = trial.checkpoint
        agent.restore(best_checkpoint.value)
        policy_model = agent.get_policy().model

        torch.save(policy_model, last_run_path/trial.trial_id/'policy_model.pt')
        models[trial.trial_id] = { "model": policy_model, "config": config }
        # weights_policy_path = LOOP_TOOL_ROOT/'loop_tool_service/models/weights/policy.pt'
        # if os.path.exists(weights_policy_path): os.remove(weights_policy_path)
        # os.symlink(last_run_path/'policy_model.pt', weights_policy_path)

    return models


def run_benchmark_greedy(env, benchmark, walk_count, step_count, search_depth, search_width):
    # breakpoint()
    env.reset(benchmark=benchmark)
    best_actions_reward_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
    print(f'Search = {best_actions_reward_str}')
    best_actions_reward = json.loads(best_actions_reward_str)
    return best_actions_reward[1] # reward

def run_benchmark_policy(env, benchmark, step_count, policy_model):
    observation, done = env.reset(benchmark=benchmark), False
    env.send_param('load_policy_model', args.policy_model)
    best_actions_reward = json.loads(env.send_param("policy_search", f'{step_count}, 5'))
    return best_actions_reward[1]

    step = 0
    while not done or step < step_count:
        logits, _ = policy_model({"obs": torch.Tensor(observation).to(device)})
        sorted_actions_q, sorted_actions = torch.sort(logits, descending=True)

        for ai, action in enumerate(sorted_actions.flatten().tolist()):
            observation, _, done, info = env.step(int(action))
            if not info['action_had_no_effect']:
                break
    
        step += 1

    return float(env.observation["flops_loop_nest_tensor"])


def run_benchmark_policy_beam(env, benchmark, step_count, policy_model):
    observation, done = env.reset(benchmark=benchmark), False
    env.send_param('load_policy_model', args.policy_model)
    best_actions_reward = json.loads(env.send_param("policy_beam_search", '100, 3'))
    return best_actions_reward[1]


# Lets define a helper function to make it easy to evaluate the agent's
# performance on a set of benchmarks.
def run_agent_on_benchmarks(policy_model, benchmarks):
    """Run agent on a list of benchmarks and return a list of cumulative rewards."""
    with make_env() as env:
        df_gflops = pd.DataFrame(columns=['bench', 'base', 'network', 'search', 'rank', 'network_actions', 'search_actions'])

        for i, benchmark in enumerate(benchmarks, start=0):
            d = {}

            d['bench'] =  str(benchmark).split('/')[-1]
            d['base'] = float(env.observation["flops_loop_nest_tensor"])

            d['greedy_0_10'] = run_benchmark_greedy(
                env=env, 
                benchmark=benchmark, 
                walk_count=10,
                step_count=args.steps,
                search_depth=0,
                search_width=10000,    
            )

            d['greedy_1_1'] = run_benchmark_greedy(
                env=env, 
                benchmark=benchmark, 
                walk_count=10,
                step_count=args.steps,
                search_depth=0,
                search_width=10000,    
            )

            d['policy'] = run_benchmark_policy(
                env=env, 
                benchmark=benchmark, 
                step_count=args.steps, 
                policy_model=policy_model
            )

            d['policy_beam'] = run_benchmark_policy(
                env=env, 
                benchmark=benchmark, 
                step_count=args.steps, 
                policy_model=policy_model
            )

  

            for key, value in d.items():
                df_gflops.at[i, key] = value

            print(f"[{i}/{len(benchmarks)}] ")
    
    return df_gflops


def plot_benchmarks(name, train, val, columns, wandb_run_id):
    global last_run_path
    # Finally lets plot our results to see how we did!
    fig, axs = plt.subplots(1, 2, figsize=(40, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f'GFlops comparison for training and test benchmarks', fontsize=16)
    if len(train): axs[0] = train.plot(x='bench', y=columns, kind='bar', ax=axs[0])
    if len(val): axs[1] = val.plot(x='bench', y=columns, kind='bar', ax=axs[1])

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(last_run_path/wandb_run_id/f"{name}.png")


def save_results(df_gflops_train, df_gflops_val, wandb_run_id):
    print("hack888")
    global last_run_path, wandb_log
    # Finally lets plot our results to see how we did!
    fig, axs = plt.subplots(2, 2, figsize=(40, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f'GFlops comparison for training and test benchmarks', fontsize=16)
    df_gflops_train_plot = df_gflops_train.sample(n = min(len(df_gflops_train), 40))
    df_gflops_val_plot = df_gflops_val.sample(n = min(len(df_gflops_val), 40))
    plot_benchmarks('benchmarks_gflops', df_gflops_train_plot, df_gflops_val_plot, columns=['base', 'network', 'search'], wandb_run_id=wandb_run_id)
    plot_benchmarks('benchmarks_rank', df_gflops_train_plot, df_gflops_val_plot, columns=['rank'], wandb_run_id=wandb_run_id)

    # Analyse results
    fig, axs = plt.subplots()
    axs.violinplot(dataset = [
        df_gflops_train['search'].astype(float) / df_gflops_train['base'].astype(float) if len(df_gflops_train) else 0,
        df_gflops_val['search'].astype(float) / df_gflops_val['base'].astype(float) if len(df_gflops_val) else 0,
        df_gflops_train['network'].astype(float) / df_gflops_train['base'].astype(float) if len(df_gflops_train) else 0,
        df_gflops_val['network'].astype(float) / df_gflops_val['base'].astype(float) if len(df_gflops_val) else 0,
    ])
    labels = ['search_train', 'search_test', 'network_train', 'network_val']
    axs.set_xticks(np.arange(1, len(labels) + 1))
    axs.set_xticklabels(labels)
    axs.set_xlim(0.25, len(labels) + 0.75)

    axs.set_title('Speedup distribution for greedy search and network approach')
    axs.yaxis.grid(True)
    axs.set_xlabel('Models')
    fig.savefig(last_run_path/wandb_run_id/"speedup_violin.png")

    # Save df
    df_gflops_all = pd.concat([df_gflops_train, df_gflops_val])
    df_gflops_all['search_speedup'] = df_gflops_all['search'].astype(float) / df_gflops_all['base'].astype(float)
    df_gflops_all['network_speedup'] = df_gflops_all['network'].astype(float) / df_gflops_all['base'].astype(float)
    df_gflops_all['final_performance'] = df_gflops_all['network'].astype(float) / df_gflops_all['search'].astype(float)

    df_gflops_all.to_csv(last_run_path/wandb_run_id/'benchmarks_gflops.csv')



def send_to_wandb(last_run_path, wandb_log):
    os.chdir(last_run_path)
    wandb_uri = f'dejang/loop_tool_agent_split/{wandb_log["run_id"]}'
    print(f'Wandb page = https://wandb.ai/{wandb_uri}')
    api = wandb.Api()
    wandb_run = api.run(wandb_uri)
    wandb_run.upload_file('benchmarks_gflops.png')
    wandb_run.upload_file('benchmarks_rank.png')
    wandb_run.upload_file('benchmarks_gflops.csv')
    wandb_run.upload_file('speedup_violin.png')
    wandb_run.upload_file('policy_model.pt')

    for key, value in wandb_log.items(): 
        wandb_run.summary[key] = value
    wandb_run.summary.update()
    


def train(config, stop_criteria, sweep_count=1, policy_model_path=''):
    print(f'Train params: ', config, stop_criteria, policy_model_path)

    register_env()
    train_benchmarks, val_benchmarks = load_datasets()

    def make_training_env(*args): 
        del args
        return CycleOverBenchmarks(make_env(), train_benchmarks)

    tune.register_env("compiler_gym", make_training_env)

    ModelCatalog.register_custom_model(
        "my_model", my_net_rl.TorchCustomModel
    )

    policy_model = torch.load(policy_model_path)


    # Evaluate agent performance on the train and validation set.
    df_gflops_train = run_agent_on_benchmarks(policy_model, train_benchmarks)
    df_gflops_val = run_agent_on_benchmarks(policy_model, val_benchmarks)

    save_results(df_gflops_train=df_gflops_train, df_gflops_val=df_gflops_val, wandb_run_id=trial_id)


    ray.shutdown()
    print("Return from train!")



    

if __name__ == '__main__':
    # init_logging(level=logging.DEBUG)
    if ray.is_initialized(): ray.shutdown()

    print(f"Running with following CLI options: {args}")

    
    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)

    
    train(config=default_config, stop_criteria=stop_criteria, sweep_count=sweep_count, policy_model_path=args.policy)
