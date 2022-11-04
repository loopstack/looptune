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
import pdb

# import gym
import numpy as np
import pickle
import os
import sys
import loop_tool as lt
import csv
import json
import random

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.service.connection import ServiceError



import loop_tool_service
from loop_tool_service.service_py.datasets import mm128_128_128
from loop_tool_service.service_py.rewards import  flops_loop_nest_reward



def register_env():
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardTensor(),
                ],
            "datasets": [
                mm128_128_128.Dataset(),
            ],
        },
    )

register_env()

def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    
    with loop_tool_service.make_env("loop_tool_env-v0") as env:
        for bench in env.datasets["benchmark://mm128_128_128-v0"]:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{bench}")
            try:
                env.reset(benchmark=bench)
                # env.send_param("timeout_sec", "1")
            except ServiceError:
                print("AGENT: Timeout Error Reset")
                continue

            
            env.send_param("print_looptree", "")


            for i in range(4):
                available_actions = json.loads(env.send_param("available_actions", ""))
                action = random.choice(available_actions)
                print(f"**********************************************************")
                print(f"Action = {action}\n")
                env.send_param("print_looptree", "")

                try:
                    observation, reward, done, info = env.step(
                        action=env.action_space.from_string(action),
                        observation_spaces=["loops_tensor"],
                        reward_spaces=["flops_loop_nest_tensor"],
                    )
                except ServiceError as e:
                    print(f"AGENT: Timeout Error Step: {e}")
                    continue
                
                print(f"{observation}\n")
                print(f"{reward}\n")
                print(f"{info}\n")
                
                

if __name__ == "__main__":
    main()
