# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function

import argparse
import os
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import (train, test, get_data_loaders,
                                             ConvNet)

from ray.rllib.agents.ppo import PPOTrainer

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

# Training settings
# parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
# parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")


from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT
from datetime import datetime


from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit
from ray import tune

import compiler_gym
import ray

def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    # We will use LLVM as our base environment. Here we specify the observation
    # space from this paper: https://arxiv.org/pdf/2003.00671.pdf and the total
    # IR instruction count as our reward space, normalized against the 
    # performance of LLVM's -Oz policy.
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )
    # Here we constrain the action space of the environment to use only a 
    # handful of command line flags from the full set. We do this to speed up
    # learning by pruning the action space by hand. This also limits the 
    # potential improvements that the agent can achieve compared to using the 
    # full action space.
    env = ConstrainedCommandline(env, flags=[
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
    ])
    # Finally, we impose a time limit on the environment so that every episode
    # for 5 steps or fewer. This is because the environment's task is continuous
    # and no action is guaranteed to result in a terminal state. Adding a time
    # limit means we don't have to worry about learning when an agent should 
    # stop, though again this limits the potential improvements that the agent
    # can achieve compared to using an unbounded maximum episode length.
    env = TimeLimit(env, max_episode_steps=5)
    return env


# Let's create an environment and print a few attributes just to check that we 
# have everything set up the way that we would like.
with make_env() as env:
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Reward space:", env.reward_space)


from itertools import islice

with make_env() as env:
  # The two datasets we will be using:
  npb = env.datasets["npb-v0"]
  chstone = env.datasets["chstone-v0"]

  # Each dataset has a `benchmarks()` method that returns an iterator over the
  # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
  # handful of benchmarks for training and validation.
  train_benchmarks = list(islice(npb.benchmarks(), 55))
  train_benchmarks, val_benchmarks = train_benchmarks[:50], train_benchmarks[50:]
  # We will use the entire chstone-v0 dataset for testing.
  test_benchmarks = list(chstone.benchmarks())

print("Number of benchmarks for training:", len(train_benchmarks))
print("Number of benchmarks for validation:", len(val_benchmarks))
print("Number of benchmarks for testing:", len(test_benchmarks))



from compiler_gym.wrappers import CycleOverBenchmarks

def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
  """Make a reinforcement learning environment that cycles over the
  set of training benchmarks in use.
  """
  del args  # Unused env_config argument passed by ray
  return CycleOverBenchmarks(make_env(), train_benchmarks)

tune.register_env("compiler_gym", make_training_env)




# Lets cycle through a few calls to reset() to demonstrate that this environment
# selects a new benchmark for each episode.
with make_training_env() as env:
  env.reset()
  print(env.benchmark)
  env.reset()
  print(env.benchmark)
  env.reset()
  print(env.benchmark)


import logging
from compiler_gym.util.logging import init_logging


if __name__ == "__main__":
    init_logging(level=logging.DEBUG)

    # ip_head and redis_passwords are set by ray cluster shell scripts
    ray_address = os.environ["RAY_ADDRESS"] if "RAY_ADDRESS" in os.environ else "auto"
    head_node_ip = os.environ["HEAD_NODE_IP"] if "HEAD_NODE_IP" in os.environ else "127.0.0.1"
    redis_password = os.environ["REDIS_PASSWORD"] if "REDIS_PASSWORD" in os.environ else "5241590000000000"
    print(ray_address, head_node_ip, redis_password)

    ray.init(address=ray_address, _node_ip_address=head_node_ip, _redis_password=redis_password)

    # sched = ASHAScheduler(metric="mean_accuracy", mode="max")

    tune.register_env("compiler_gym", make_training_env)
    
    analysis = tune.run(
        PPOTrainer,
        checkpoint_at_end=True,
        stop={
            "episodes_total": 500,
        },
        config={
            "seed": 0xCC,
            "num_workers": 1,
            # Specify the environment to use, where "compiler_gym" is the name we 
            # passed to tune.register_env().
            "env": "compiler_gym",
            # Reduce the size of the batch/trajectory lengths to match our short 
            # training run.
            "rollout_fragment_length": 5,
            "train_batch_size": 5,
            "sgd_minibatch_size": 5,
        },
        callbacks=[ WandbLoggerCallback(
                                project="loop_tool_agent",
                                api_key_file=str(LOOP_TOOL_ROOT) + "/wandb_key.txt",
                                log_config=False)]
    )
                        
    
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode="max"))
