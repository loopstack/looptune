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

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import to_pydot

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
from service_py.rewards import runtime_reward, flops_reward, flops_loop_nest_reward

import loop_tool_service.models.qlearningAgents as q_agents

def register_env():
    register(
        id="loop_tool-v0",
        entry_point=loop_tool_service.LoopToolCompilerEnv,
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
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

    
    state = None
    state_prev = None
    current_flops = None


    with loop_tool_service.make_env("loop_tool-v0") as env:
        agent = q_agents.QLearningAgent(
            actionSpace=env.env.action_space,
            numTraining=900, 
            exploration=0.5, 
            learning_rate=0.2, 
            discount=0.8,
        )

        bench = "benchmark://loop_tool_simple-v0/mm256"

        try:
            env.reset(benchmark=bench)
            # env.send_param("timeout_sec", "1")
        except ServiceError as e:
            print(f"AGENT: Timeout Error Reset: {e}")
            return

        available_actions = json.loads(env.send_param("available_actions", ""))
        observation = env.observation["ir_networkx"]
        state_nx = pickle.loads(observation)
        state = q_agents.State(hash=nx.weisfeiler_lehman_graph_hash(state_nx, node_attr='feature'),
                               string=to_pydot(state_nx).to_string())
        
        start_reward = env.reward["flops_loop_nest"]
        state_start = state

        agent.registerInitialState(state=state)

        for i in range(1000):
            # pdb.set_trace()
            state_prev = state
            action = agent.getAction(state, available_actions)

            print(f"**************************** {i} ******************************")
            print(f"Action = {env.action_space.to_string(action)}\n")
            # pdb.set_trace()

            try:
                observation, rewards, done, info = env.step(
                    action=action,
                    observation_spaces=["ir_networkx"],
                    reward_spaces=["flops_loop_nest"],
                )
            except ServiceError as e:
                print(f"AGENT: Timeout Error Step: {e}")
                continue
            except ValueError:
                pdb.set_trace()
                pass
            
            env.send_param("print_looptree", "")

            # available_actions = info[""]
            available_actions = json.loads(env.send_param("available_actions", ""))
            print(f"Available_actions = {available_actions}")
            state_nx = pickle.loads(observation[0])
            state = q_agents.State(hash=nx.weisfeiler_lehman_graph_hash(state_nx, node_attr='feature'),
                                   string=to_pydot(state_nx).to_string())
        

            print(f"{rewards}\n")
            print(f"{info}\n")


            agent.observeTransition(state_prev, action, state, rewards[0])
            # agent.update(state_prev, action, state, rewards[0])
            print(agent.Q)


            # pdb.set_trace()
            current_flops = env.observation["flops_loop_nest"]
            print(f"Current speed = {current_flops/1e9} GFLOPS")
            print(state.string)
            agent.stopEpisode()

        agent.plot_history()
        pdb.set_trace()
        agent.plot_policy()
        pdb.set_trace()

        print(f"====================================================================")
        print(f"Start speed = {start_reward/1e9} GFLOPS")
        print(f"Final speed = {current_flops/1e9} GFLOPS")

if __name__ == "__main__":
    main()
