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

import loop_tool_service.models.qlearningAgents as q_agents

def register_env():
    register(
        id="loop_tool-v0",
        entry_point=loop_tool_service.LoopToolCompilerEnv,
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                runtime_reward.Reward(),
                flops_reward.Reward(),
                # flops_loop_nest_reward.Reward(),
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

    
    state = lt.Tensor()
    state_prev = lt.Tensor()

    with loop_tool_service.make_env("loop_tool-v0") as env:
        agent = q_agents.QLearningAgent(
            actionSpace=env.env.action_space,
            numTraining=1000, 
            exploration=0.5, 
            learning_rate=0.2, 
            discount=0.8,
        )

        bench = "benchmark://loop_tool_simple-v0/muladd"

        try:
            env.reset(benchmark=bench)
            # env.send_param("timeout_sec", "1")
        except ServiceError:
            print("AGENT: Timeout Error Reset")
            return

        available_actions = json.loads(env.send_param("available_actions", ""))
        observation = env.observation["loop_tree_ir"]
        state.set(lt.deserialize(observation))
        state_start = state

        
        for i in range(1005):
            # pdb.set_trace()
            state_prev = state
            action = agent.getAction(state, available_actions)

            print(f"**************************** {i} ******************************")
            print(f"Action = {env.action_space.to_string(action)}\n")
            try:
                observation, rewards, done, info = env.step(
                    action=action,
                    observation_spaces=["loop_tree_ir"],
                    reward_spaces=["flops"],
                )
            except ServiceError:
                print("AGENT: Timeout Error Step")
                continue
            except ValueError:
                pdb.set_trace()
                pass
            # available_actions = info[""]
            available_actions = json.loads(env.send_param("available_actions", ""))
            print(f"Available_actions = {available_actions}")
            state.set(lt.deserialize(observation[0]))
            print(state.loop_tree)
            print(f"{rewards}\n")
            print(f"{info}\n")

            agent.update(state_prev, action, state, rewards[0])
            print(agent.Q)


            # pdb.set_trace()
            print(f"Current speed = {state.loop_tree.flops() / state.loop_tree.eval() / 1e9} GFLOPS")



        print(f"====================================================================")
        print(f"Start speed = {state_start.loop_tree.flops() / state_start.loop_tree.eval() / 1e9} GFLOPS")
        print(f"Final speed = {state.loop_tree.flops() / state.loop_tree.eval() / 1e9} GFLOPS")

if __name__ == "__main__":
    main()
