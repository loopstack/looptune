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

from loop_tool_service.service_py.datasets import loop_tool_dataset, loop_tool_test_dataset

from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
import loop_tool_service.models.rllib.my_net_rl as my_net_rl


import wandb
from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT


# Run this with: 
# python launcher/slurm_launch.py -e launcher/exp.yaml -n 1 -t 3:00   ### slurm_launch.py internaly calls rllib_torch.py
# python

# wandb.tensorboard.patch(root_logdir="...")
# wandb.init(project="loop_tool_agent", entity="dejang")


# tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()



last_run_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

stop_criteria = {'training_iteration': 1}
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
wandb_log = {}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--policy",  type=str, nargs='?', const=f'{last_run_path}/policy_model.pt', default='', help="Load policy network."
)
parser.add_argument(
    "--sweep",  type=int, nargs='?', const=2, default=0, help="Run with wandb sweeps"
)
parser.add_argument(
    "--slurm", 
    default=False, 
    action="store_true",
    help="Run on slurm"
)
parser.add_argument(
    "--iter", type=int, default=20, help="Number of iterations to train."
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

args = parser.parse_args()




def register_env():
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardTensor(),
                # flops_loop_nest_reward.AbsoluteRewardTensor(),
                ],
            "datasets": [
                # loop_tool_dataset.Dataset(),
                loop_tool_test_dataset.Dataset()
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

    env = TimeLimit(env, max_episode_steps=10) # <<<< Must be here
    return env


def load_datasets(env=None):

    with make_env() as env:
        # The two datasets we will be using:
        lt_dataset = env.datasets["benchmark://loop_tool_test-v0"]
        benchmarks = random.sample(list(lt_dataset.benchmarks()), min(len(lt_dataset), args.size) )

        train_perc = 0.8
        train_size = int(train_perc * len(benchmarks))
        test_size = len(benchmarks) - train_size
        train_benchmarks, val_benchmarks = torch.utils.data.random_split(benchmarks, [train_size, test_size])

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


# Lets define a helper function to make it easy to evaluate the agent's
# performance on a set of benchmarks.
def run_agent_on_benchmarks(policy_model, benchmarks):
    """Run agent on a list of benchmarks and return a list of cumulative rewards."""
    with make_env() as env:
        df_gflops = pd.DataFrame(np.zeros((len(benchmarks), 5)), columns=['bench', 'base', 'network', 'search', 'rank'])

        for i, benchmark in enumerate(benchmarks, start=0):
            observation, done = env.reset(benchmark=benchmark), False
            step_count = 0
            df_gflops.loc[i, 'bench'] =  str(benchmark).split('/')[-1]
            df_gflops.loc[i, 'base'] = env.observation["flops_loop_nest_tensor"]

            # breakpoint()
            
            while not done:
                logits, _ = policy_model({"obs": torch.Tensor(observation).to(device)})
                sorted_actions_q, sorted_actions = torch.sort(logits, descending=True)

                # for q, a in zip(sorted_actions_q.flatten().tolist(), sorted_actions.flatten().tolist()):
                #     print(env.action_space.to_string(a), q)

                chosen_rank = []
                for ai, action in enumerate(sorted_actions.flatten().tolist()):
                    observation, _, done, info = env.step(int(action))
                    if not info['action_had_no_effect']:
                        chosen_rank.append(ai)
                        break
            
                flops = env.observation["flops_loop_nest_tensor"]
                df_gflops.loc[i, 'network'] = flops
                step_count += 1
                # print(f'{step_count}. Flops = {flops}, Actions = {[ env.action_space.to_string(a) for a in env.actions]}')

            df_gflops.loc[i, 'rank'] = np.mean(chosen_rank)

            env.send_param("print_looptree", "")
            print(f'My network = {flops}, {env.send_param("actions", "")}')

            walk_count = 1
            step_count = 20
            search_depth=1
            search_width = 10000
            # breakpoint()
            env.reset(benchmark=benchmark)
            best_actions_reward_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
            print(f'Search = {best_actions_reward_str}')
            best_actions_reward = json.loads(best_actions_reward_str)
            df_gflops.loc[i, 'search'] = best_actions_reward[1]

            print(f"[{i}/{len(benchmarks)}] ")

    
    return df_gflops


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
    

def plot_benchmarks(name, train, val, columns, wandb_run_id):
    global last_run_path
    # Finally lets plot our results to see how we did!
    fig, axs = plt.subplots(1, 2, figsize=(40, 5), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle(f'GFlops comparison for training and test benchmarks', fontsize=16)
    axs[0] = train.plot(x='bench', y=columns, kind='bar', ax=axs[0])
    axs[1] = val.plot(x='bench', y=columns, kind='bar', ax=axs[1])

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(last_run_path/wandb_run_id/f"{name}.png")


def plot_results(df_gflops_train, df_gflops_val, wandb_run_id, config):
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
        df_gflops_train['search'] / df_gflops_train['base'],
        df_gflops_val['search'] / df_gflops_val['base'],
        df_gflops_train['network'] / df_gflops_train['base'],
        df_gflops_val['network'] / df_gflops_val['base'],
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
    df_gflops_all['search_speedup'] = df_gflops_all['search'] / df_gflops_all['base']
    df_gflops_all['network_speedup'] = df_gflops_all['network'] / df_gflops_all['base']
    df_gflops_all['final_performance'] = df_gflops_all['network'] / df_gflops_all['search']

    df_gflops_all.to_csv(last_run_path/wandb_run_id/'benchmarks_gflops.csv')

    wandb_log['group_id'] = wandb_run_id.split('_')[0]
    wandb_log['run_id'] = wandb_run_id
    wandb_log['algorithm'] = 'PPO'


    if 'fcnet_hiddens' in config['model']:
        wandb_log['layers_num'] = len(config['model']['fcnet_hiddens'])
        wandb_log['layers_width'] = config['model']['fcnet_hiddens'][0]
    
    wandb_log['final_performance'] = np.mean(df_gflops_val['network'] / df_gflops_val['search'])
    wandb_log['avg_search_base_speedup'] = np.mean(df_gflops_val['search'] / df_gflops_val['base'])
    wandb_log['avg_network_base_speedup'] = np.mean(df_gflops_val['network'] / df_gflops_val['base'])
    wandb_log['rank'] = np.mean(df_gflops_val['rank'])

    wandb_log_path = last_run_path/wandb_run_id/"wandb_log.json"
    with open(wandb_log_path, "w") as f: json.dump(wandb_log, f)

    # Send results to wandb server
    if wandb_run_id not in [None, '']:
        send_to_wandb(last_run_path/wandb_run_id, wandb_log)

    print(f'Final performance = {wandb_log["final_performance"]}')
    print(f'avg_search_base_speedup = {wandb_log["avg_search_base_speedup"]}')
    print(f'avg_network_base_speedup = {wandb_log["avg_network_base_speedup"]}')


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

    print("hhh1______________________")
    if policy_model_path == '':
        models = train_agent(
            config=config, 
            stop_criteria=stop_criteria, 
            sweep_count=sweep_count, 
        )
    else:
        models = {}
        models[''] = { "model": torch.load(policy_model_path), "config": config }



    for trial_id, policy_model in models.items():
        # Evaluate agent performance on the train and validation set.
        df_gflops_train = run_agent_on_benchmarks(policy_model["model"], train_benchmarks)
        df_gflops_val = run_agent_on_benchmarks(policy_model["model"], val_benchmarks)

        plot_results(df_gflops_train, df_gflops_val, trial_id, policy_model["config"])

    ray.shutdown()
    print("Return from train!")


def update_default_config(sweep_config=None):
    for key, val in default_config.items():
        if key in sweep_config:
            if type(val) == dict:
                val.update(sweep_config[key])
            else:
                default_config[key] = sweep_config[key]

    return default_config
    

if __name__ == '__main__':
    # init_logging(level=logging.DEBUG)
    if ray.is_initialized(): ray.shutdown()

    print(f"Running with following CLI options: {args}")

    stop_criteria['training_iteration'] = 2 if args.debug or False else args.iter
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

    if sweep_count and args.policy != '':
        hiddens_layers = [4, 8, 20]
        hiddens_width = [100, 500, 1000]
        sweep_config = {
            'lr': tune.uniform(1e-4, 1e-7),
            "gamma": tune.uniform(0.5, 0.99),
            'model': {
                "fcnet_hiddens": tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]),
            },

        }
        default_config = update_default_config(sweep_config)

    
    train(config=default_config, stop_criteria=stop_criteria, sweep_count=sweep_count, policy_model_path=args.policy)
