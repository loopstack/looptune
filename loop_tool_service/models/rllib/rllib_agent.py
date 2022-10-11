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

from loop_tool_service.service_py.datasets import mm128_128_128

from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
import loop_tool_service.models.rllib.my_net_rl as my_net_rl


import wandb
from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT


# Run this with: 
# python launcher/slurm_launch.py -e launcher/exp.yaml -n 1 -t 3:00   ### slurm_launch.py internaly calls rllib_torch.py
# python

torch, nn = try_import_torch()


def make_env():
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )

    env = TimeLimit(env, max_episode_steps=20) # <<<< Must be here
    return env

class RLlibAgent:
    def __init__(self, use_wandb=True) -> None:
        self.env = make_env()
        self.train_iter = 20
        self.last_run_path=LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_wandb = use_wandb
        self.wandb_log = {}


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
            self.wandb_log['dataset'] = dataset.name
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
        self.wandb_log['algorithm'] = algorithm._name
        self.wandb_log['actions'] = ",".join(self.env.action_space.names)

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
            os.makedirs(self.last_run_path/trial.trial_id)
            config = trial.config
            config["explore"] = False
            agent = algorithm(
                env="compiler_gym",
                config=config
            )
            best_checkpoint = trial.checkpoint
            agent.restore(best_checkpoint.value)
            policy_model = agent.get_policy().model
            models[trial.trial_id] = { 
                "policy_path": self.last_run_path/trial.trial_id/'policy_model.pt', 
                "config": config 
            }
            torch.save(policy_model,  models[trial.trial_id]["policy_path"])
            # weights_policy_path = LOOP_TOOL_ROOT/'loop_tool_service/models/weights/policy.pt'
            # if os.path.exists(weights_policy_path): os.remove(weights_policy_path)
            # os.symlink(self.last_run_path/'policy_model.pt', weights_policy_path)

        return models


    ''' TODO: Dejan 
        create class Evaluator -> eval(benchmarks) that runs custom searches creates
        charts, and returns paths of charts.
        Evaluator.loads(cost/policy model)
        This is used for both experiments/search and rllib_torch_ppo for evaluation
    '''

    # Lets define a helper function to make it easy to evaluate the agent's
    # performance on a set of benchmarks.
    def run_agent_on_benchmarks(self, policy_path, benchmarks):
        """Run agent on a list of benchmarks and return a list of cumulative rewards."""
        df_gflops = pd.DataFrame(columns=['bench', 'base', 'network', 'search', 'rank', 'network_actions', 'search_actions'])

        for i, benchmark in enumerate(benchmarks, start=0):
            d = {}
            d_network = self.run_benchmark_network(env=self.env, benchmark=benchmark, policy_path=policy_path)
            d.update(d_network)
            
            d_search = self.run_benchmark_search(env=self.env, benchmark=benchmark)
            d.update(d_search)

            for key, value in d.items():
                df_gflops.at[i, key] = value

            print(f"[{i}/{len(benchmarks)}] ")
        
        return df_gflops


    def run_benchmark_network(self, env, benchmark, policy_path, train_iter):
        d = {}
        search_width=1000000
        env.reset(benchmark=benchmark)
        d['bench'] =  str(benchmark).split('/')[-1]
        d['base'] = float(env.observation["flops_loop_nest_tensor"])

        env.send_param('load_policy_model', str(policy_path))
        actions_reward = json.loads(env.send_param("beam_search", f'{train_iter}, {search_width}, loop_nest'))

        d['network_actions'] = actions_reward[0]
        d['network'] = actions_reward[1]
        
        print(f'My network = {actions_reward}')
        return d


    def run_benchmark_search(self, env, benchmark, train_iter):
        d = {}
        walk_count = 1
        search_depth=1
        search_width = 10000
        # breakpoint()
        env.reset(benchmark=benchmark)
        actions_reward = json.loads(env.send_param("beam_search", f'{train_iter}, {search_width}, loop_nest'))
        best_actions_reward_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
        print(f'Search = {best_actions_reward_str}')
        best_actions_reward = json.loads(best_actions_reward_str)
        d['search_actions'] = best_actions_reward[0]
        d['search'] = best_actions_reward[1]
        return d

    def save_results(self, trial_id, policy_model, df_gflops_train, df_gflops_val):
        print("hack888")

        self.plot_results(trial_id, df_gflops_train, df_gflops_val)

        df_gflops_all = pd.concat([df_gflops_train, df_gflops_val])
        df_gflops_all['search_speedup'] = df_gflops_all['search'].astype(float) / df_gflops_all['base'].astype(float)
        df_gflops_all['network_speedup'] = df_gflops_all['network'].astype(float) / df_gflops_all['base'].astype(float)
        df_gflops_all['final_performance'] = df_gflops_all['network'].astype(float) / df_gflops_all['search'].astype(float)
        df_gflops_all.to_csv(self.last_run_path/trial_id/'benchmarks_gflops.csv')


        if self.use_wandb:
            self.finalize_wandb(
                wandb_run_id=trial_id, 
                config=policy_model["config"],
                df_gflops_train=df_gflops_train, 
                df_gflops_val=df_gflops_val, 
            )

    def plot_results(self, trial_id, df_gflops_train, df_gflops_val):
        # Finally lets plot our results to see how we did!
        fig, axs = plt.subplots(2, 2, figsize=(40, 5), gridspec_kw={'width_ratios': [1, 1]})
        fig.suptitle(f'GFlops comparison for training and test benchmarks', fontsize=16)
        df_gflops_train_plot = df_gflops_train.sample(n = min(len(df_gflops_train), 40))
        df_gflops_val_plot = df_gflops_val.sample(n = min(len(df_gflops_val), 40))
        self.plot_benchmarks('benchmarks_gflops', df_gflops_train_plot, df_gflops_val_plot, columns=['base', 'network', 'search'], trial_id=trial_id)
        self.plot_benchmarks('benchmarks_rank', df_gflops_train_plot, df_gflops_val_plot, columns=['rank'], trial_id=trial_id)

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
        fig.savefig(self.last_run_path/trial_id/"speedup_violin.png")


    def plot_benchmarks(self, name, train, val, columns, trial_id):
        # Finally lets plot our results to see how we did!
        fig, axs = plt.subplots(1, 2, figsize=(40, 5), gridspec_kw={'width_ratios': [1, 1]})
        fig.suptitle(f'GFlops comparison for training and test benchmarks', fontsize=16)
        if len(train): axs[0] = train.plot(x='bench', y=columns, kind='bar', ax=axs[0])
        if len(val): axs[1] = val.plot(x='bench', y=columns, kind='bar', ax=axs[1])

        fig.autofmt_xdate()
        plt.tight_layout()
        fig.savefig(self.last_run_path/trial_id/f"{name}.png")



    def finalize_wandb(self, wandb_run_id, config, df_gflops_train, df_gflops_val):
        # Save df
        self.update_wandb(df_gflops_train, prefix='train_')
        self.update_wandb(df_gflops_val, prefix='')
        print(f'Final performance = {self.wandb_log["final_performance"]}')
        print(f'avg_search_base_speedup = {self.wandb_log["avg_search_base_speedup"]}')
        print(f'avg_network_base_speedup = {self.wandb_log["avg_network_base_speedup"]}')

        self.wandb_log['group_id'] = wandb_run_id.split('_')[0]
        self.wandb_log['run_id'] = wandb_run_id

        if 'fcnet_hiddens' in config['model']:
            self.wandb_log['layers_num'] = len(config['model']['fcnet_hiddens'])
            self.wandb_log['layers_width'] = config['model']['fcnet_hiddens'][0]
            
        wandb_log_path = self.last_run_path/wandb_run_id/"wandb_log.json"
        with open(wandb_log_path, "w") as f: json.dump(self.wandb_log, f)

        # Send results to wandb server
        self.send_to_wandb(wandb_run_id)


    def update_wandb(self, df_gflops_val, prefix):
        self.wandb_log[f'{prefix}final_performance'] = float(np.mean(df_gflops_val['network'] / df_gflops_val['search']))
        self.wandb_log[f'{prefix}avg_search_base_speedup'] = float(np.mean(df_gflops_val['search'] / df_gflops_val['base']))
        self.wandb_log[f'{prefix}avg_network_base_speedup'] = float(np.mean(df_gflops_val['network'] / df_gflops_val['base']))
        self.wandb_log[f'{prefix}rank'] = float(np.mean(df_gflops_val['rank']))
        self.wandb_log[f'{prefix}data_size'] = float(len(df_gflops_val))
        self.wandb_log[f'{prefix}search_actions_num'] = float(np.mean(df_gflops_val['search_actions'].str.len()))
        self.wandb_log[f'{prefix}network_actions_num'] = float(np.mean(df_gflops_val['network_actions'].str.len()))


    def send_to_wandb(self, wandb_run_id):
        os.chdir(self.last_run_path/wandb_run_id)
        wandb_uri = f'dejang/loop_tool_agent_split/{self.wandb_log["run_id"]}'
        print(f'Wandb page = https://wandb.ai/{wandb_uri}')
        api = wandb.Api()
        wandb_run = api.run(wandb_uri)
        wandb_run.upload_file('policy_model.pt')
        wandb_run.upload_file('benchmarks_gflops.png')
        wandb_run.upload_file('benchmarks_rank.png')
        wandb_run.upload_file('benchmarks_gflops.csv')
        wandb_run.upload_file('speedup_violin.png')

        for key, value in self.wandb_log.items(): 
            wandb_run.summary[key] = value
        wandb_run.summary.update()
       




#################################################################################


def train(config, train_iter, sweep_count=1, policy_model_path=''):
    print(f'Train params: ', config, train_iter, policy_model_path)
    
    agent = RLlibAgent(
        use_wandb=True
    )

    train_benchmarks, val_benchmarks = agent.load_datasets(
        datasets=['benchmark://mm8_16_128_128-v0'],
        data_size=10000
    )

    models = agent.train(
        algorithm=PPOTrainer, 
        config=config, 
        train_iter=train_iter, 
        sweep_count=sweep_count
    )


    for trial_id, policy_model in models.items():
        # Evaluate agent performance on the train and validation set.
        df_gflops_train = agent.run_agent_on_benchmarks(policy_model["policy_path"], train_benchmarks, train_iter=train_iter)
        df_gflops_val = agent.run_agent_on_benchmarks(policy_model["policy_path"], val_benchmarks, train_iter=train_iter)

        agent.save_results(trial_id=trial_id, policy_model=policy_model, df_gflops_train=df_gflops_train, df_gflops_val=df_gflops_val)




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

    
    train(config=default_config, train_iter=args.iter, sweep_count=sweep_count, policy_model_path=args.policy)

    ray.shutdown()
    print("Return from train!")