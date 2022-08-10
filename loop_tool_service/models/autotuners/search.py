# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Perform a random walk of the action space of a CompilerGym environment.

Example usage:

    # Run a random walk on cBench example program using perf runtime reward.
    $ python demo_random_walk.py --walk_count=2 --step_count=4 --data_set=benchmark://poj104-small-v0 --reward=perf_tensor --observation=perf_tensor

"""
from importlib.metadata import entry_points
import random
import pdb
import uuid
import os
import sys

from loop_tool_service.paths import LOOP_TOOL_ROOT


from compiler_gym.util.registration import register


from absl import app, flags
from tqdm import tqdm
from joblib import Parallel, delayed

flags.DEFINE_integer("bench_count", 2, "The number of benchmarks.")
flags.DEFINE_integer("search_depth", 1, "How deep you go in search before you decide.")
flags.DEFINE_integer("search_width", 1000, "The number action you expand every level of the search.")
flags.DEFINE_integer("walk_count", 5, "The number of walks.")
flags.DEFINE_integer("step_count", 6, "The number of steps.")
flags.DEFINE_string("reward", "flops_loop_nest", "Reward.")
flags.DEFINE_string("data_set", "benchmark://loop_tool_test-v0", "Data set.")
flags.DEFINE_string("model_path",  f"{str(LOOP_TOOL_ROOT)}/loop_tool_service/models/weights/model_cost.pth", "If you want to search based on model")

FLAGS = flags.FLAGS


import loop_tool_service
from loop_tool_service.models.autotuners.greedy_walk import GreedyWalker
from loop_tool_service.service_py.rewards import flops_loop_nest_reward
from loop_tool_service.service_py.datasets import loop_tool_test_dataset


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
                loop_tool_test_dataset.Dataset(),
            ],
        },
    )



import logging
def main(argv):
    """Main entry point."""
    assert len(argv) == 1, f"Unrecognized flags: {argv[1:]}"
    # This two lines try to suppress logging to stdout.
    logging.basicConfig(level=logging.CRITICAL, force=True)
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

    register_env()

    with loop_tool_service.make_env("loop_tool-v0") as env:

        data_set = env.datasets[FLAGS.data_set]
        i = 0
        for bench in tqdm(data_set, total=min(len(data_set), FLAGS.bench_count)):
            if i == FLAGS.bench_count: break
            print(bench)
            env.reset()

            if FLAGS.model_path != '':
                env.send_param('load_cost_model', FLAGS.model_path)

            env.send_param("greedy_search", f'{FLAGS.walk_count}, {FLAGS.step_count}, {FLAGS.search_depth}, {FLAGS.search_width}')
            i += 1


        
if __name__ == "__main__":
    app.run(main)
