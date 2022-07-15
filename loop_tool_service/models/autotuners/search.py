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




from compiler_gym.util.registration import register


from absl import app, flags

from joblib import Parallel, delayed

flags.DEFINE_integer("seek_count", 10, "The number seek steps before you decide.")
flags.DEFINE_integer("walk_count", 10, "The number of walks.")
flags.DEFINE_integer("step_count", 20, "The number of steps.")
flags.DEFINE_string("data_set", "benchmark://poj104-v0", "Data set.")

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
                flops_loop_nest_reward.RewardTensor(),
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
        walker = GreedyWalker(env=env, 
                              dataset_uri = FLAGS.data_set,
                              observation=FLAGS.observation,
                              reward=FLAGS.reward,  
                              walk_count=max(1, FLAGS.walk_count),
                              step_count=max(1, FLAGS.step_count)
                              )


        walker.run()

        # walker = RandomWalker(env=env, 
        #                       dataset_uri = FLAGS.data_set,
        #                       observation=FLAGS.observation,
        #                       reward=FLAGS.reward,
        #                       walk_count=max(1, FLAGS.walk_count),
        #                       step_count=max(1, FLAGS.step_count),                               
        #                       )


        # walker.run()
        
if __name__ == "__main__":
    app.run(main)
