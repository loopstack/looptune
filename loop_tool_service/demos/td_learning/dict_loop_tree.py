# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Goal:
    This script demonstrates temoporal-difference learning algorithm with LoopTool
    environment, loop_tree observation and flops_loop_nest reward. Results of this 
    script will be visualized in demo.png inside demos directory.

Usage:
    $ python loop_tree.py

It is equivalent in behavior to the demo.py script in this directory.
"""
import logging
from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register


import loop_tool_service
from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import flops_loop_nest_reward

import loop_tool_service.models.q_agents.qAgentsDict as q_agents

def register_env():
    register(
        id="loop_tool-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardScalar(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
            ],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    bench = "benchmark://loop_tool_simple-v0/mm128"

    with loop_tool_service.make_env("loop_tool-v0") as env:
        agent = q_agents.QAgentLoopTree(
            env=env,
            bench=bench,
            observation = "loop_tree",
            reward="flops_loop_nest",
            numTraining=100, 
            numTest=4,
            exploration=0.7, 
            learning_rate=0.8, 
            discount=0.01,
        )
        agent.train()
        agent.test()


if __name__ == "__main__":
    main()
