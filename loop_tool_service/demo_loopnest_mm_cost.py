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
from loop_tool_service.service_py.rewards import flops_loop_nest_reward
# import gym
import numpy as np
import pickle
import os
import sys
import loop_tool as lt
import csv
import json

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.service.connection import ServiceError


import loop_tool_service


from service_py.datasets import loop_tool_dataset
from service_py.rewards import runtime_reward, flops_reward

import loop_tool_service.models.costAgentsNN as cost_agent

def register_env():
    register(
        id="loop_tool-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv", #loop_tool_service.LoopToolCompilerEnv,
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.AbsoluteRewardTensor(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
            ],
        },
    )

import gym
import compiler_gym

def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    bench = "benchmark://loop_tool_simple-v0/simple"
    with compiler_gym.make("loop_tool-v0") as env:
        breakpoint()
        agent = cost_agent.CostAgent(
            env=env,
            bench=bench,
            observation = "stride_tensor",
            reward="flops_loop_nest_tensor",
            numTraining=100, 
            numTest=4,
            exploration=1, 
            learning_rate=0.8, 
            discount=0.9,
        )
        agent.train()
        breakpoint()
        agent.test()


if __name__ == "__main__":
    main()