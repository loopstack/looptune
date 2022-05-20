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

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.datasets.uri import BenchmarkUri
from compiler_gym.spaces import Reward
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path
from compiler_gym.service.connection import ServiceError

from compiler_gym.envs.llvm.datasets import (
    AnghaBenchDataset,
    BlasDataset,
    CBenchDataset,
    CBenchLegacyDataset,
    CBenchLegacyDataset2,
    CHStoneDataset,
    CsmithDataset,
    NPBDataset,
)

import loop_tool_service


from service_py.datasets import loop_tool_dataset
from service_py.rewards import runtime_reward, flops_reward



def register_env():
    register(
        id="loop_tool-v0",
        entry_point=loop_tool_service.LoopToolCompilerEnv,
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                runtime_reward.Reward(),
                flops_reward.Reward(),
                flops_loop_nest_reward.Reward(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
                CBenchDataset(site_data_path("llvm-v0"))],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    actions = ["down", "split_16", "down", "down", "split_16", "down", "split_4",  "down", "vectorize"]

    with loop_tool_service.make_env("loop_tool-v0") as env:
        for bench in env.datasets["benchmark://loop_tool_simple-v0"]:
            try:
                env.reset(benchmark=bench)
                # env.send_param("timeout_sec", "1")
            except ServiceError:
                print("AGENT: Timeout Error Reset")
                continue

            for action in actions:
                print(f"**********************************************************")
                print(f"Action = {action}\n")
                try:
                    observation, reward, done, info = env.step(
                        action=env.action_space.from_string(action),
                        observation_spaces=["loop_tree_ir"],
                        reward_spaces=["flops", "flops_loop_nest"],
                    )
                except ServiceError:
                    print("AGENT: Timeout Error Step")
                    continue
            

                print(f"{reward}\n")
                print(f"{info}\n")

                try:
                    tensor = lt.Tensor()
                    tensor.set(lt.deserialize(observation[0]))
                    print(tensor.loop_tree)
                except:
                    print(f"{observation}\n")


                pdb.set_trace()


if __name__ == "__main__":
    main()
