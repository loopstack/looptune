'''
In this experiment we want to compare time and accuracy of cost model to getting ground truth with FLOPS() method
'''
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script demonstrates how the Python example service without needing
to use the bazel build system. Usage:

    $ python example_compiler_gym_service/demo_without_bazel.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
from pathlib import Path
from typing import Iterable
import time

import numpy as np
import json
from os.path import exists



import loop_tool_service

import argparse
import json
import torch
from tqdm import tqdm

import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.wrappers import TimeLimit
import wandb
import shutil

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

import loop_tool_service
# from loop_tool_service.models.rllib.rllib_torch import load_datasets, make_env
import loop_tool_service.models.rllib.my_net_rl as my_net_rl
from loop_tool_service.paths import LOOP_TOOL_ROOT

last_run_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
policy_paths = list(Path(last_run_path).glob('**/policy_model.pt'))
policy_path = str(policy_paths[0]) if len(policy_paths) else ""

# Training settings
parser = argparse.ArgumentParser(description="LoopTool Optimizer")
parser.add_argument("--policy",  type=str, nargs='?', const=policy_path , default='', help="Load policy network.")
parser.add_argument("--cost", type=str, nargs='?', const=f"{str(LOOP_TOOL_ROOT)}/loop_tool_service/models/weights/model_cost.pth", default='', help="Path to the RLlib optimized network.")
args = parser.parse_args()
print(args)


weights_path = LOOP_TOOL_ROOT/"loop_tool_service/models/weights"


def resolve_policy(policy_path):
    if policy_path == '' or exists(policy_path):
        return str(policy_path)
    try:
        wandb.restore('policy_model.pt', run_path=policy_path)
    except:
        print('Policy not found')
        exit(1)
        
    shutil.move("policy_model.pt", weights_path/'policy.pt')
    return str(weights_path/'policy.pt')



def resolve_cost(cost_path):
    if cost_path == '' or exists(cost_path):
        return str(cost_path)
    try:
        wandb.restore('cost_model.pt', run_path=cost_path)
    except:
        print('Cost path not found')
        exit(1)
        
    shutil.move("cost_model.pt", weights_path/'cost.pt')
    return str(weights_path/'cost.pt')



if __name__ == '__main__':
    
    print(args)
    # register_env()
    

    with loop_tool_service.make_env("loop_tool_env-v0") as env:
        i = 0
        for bench in env.datasets["benchmark://loop_tool_simple-v0"]: 
            print(bench)
            env.reset(benchmark=bench)
            
            if args.cost != '':
                env.send_param('load_cost_model', resolve_cost(args.cost))

            time0 = time.time()
            gflops_ln = env.observation["flops_loop_nest"]
            time1 = time.time()
            gflops_cost = env.observation["gflops_cost"]
            time2 = time.time()


            print(f"{gflops_ln} GFLOPS")
            env.send_param('print_looptree', args.cost)

            print('Times:')
            print(f"gflops_ln:\t{time1 - time0}\n")
            print(f"gflops_cost:\t{time2 - time1}\n")
                
                