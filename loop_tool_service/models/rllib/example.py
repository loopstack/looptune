import gym
from gym.spaces import Discrete, Box
import numpy as np
import random

from ray.rllib.env.env_context import EnvContext


class MysteriousCorridor(gym.Env):
    """Example of a custom env in which you walk down a mysterious corridor.

    You can configure the reward of the destination state via the env config.

    A mysterious corridor has 7 cells and looks like this:
    -------------------------------------------
    |  1  |  2  |  3  |  4  |  3  |  5  |  6  |
    -------------------------------------------
    You always start from state 1 (left most) or 6 (right most).
    The goal is to get to the destination state 4 (in the middle).
    There are only 2 actions, 0 means go left, 1 means go right.
    """

    def __init__(self, config: EnvContext):
        self.seed(random.randint(0, 1000))

        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 6.0, shape=(1, ), dtype=np.float32)
        self.reward = config["reward"]

        self.reset()

    def reset(self):
        # cur_pos is the actual postion of the player. not the state a player
        # sees from outside of the environemtn.
        # E.g., when cur_pos is 1, the returned state is 3.
        # Start from either side of the corridor, 0 or 4.
        self.cur_pos = random.choice([0, 6])
        return [self.cur_pos]

    def _pos_to_state(self, pos):
        ptos = [1, 2, 3, 4, 3, 5, 6]
        return ptos[pos]

    def step(self, action):
        assert action in [0, 1], action

        if action == 0:
            self.cur_pos = max(0, self.cur_pos - 1)
        if action == 1:
            self.cur_pos = min(6, self.cur_pos + 1)

        done = (self.cur_pos == 3)
        reward = self.reward if done else -0.1

        return [self._pos_to_state(self.cur_pos)], reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)

    def render(self):
        def symbol(i):
            if i == self.cur_pos:
                return "o"
            elif i == 3:
                return "x"
            elif i == 2 or i == 4:
                return "_"
            else:
                return " "
        return "| " + " | ".join([symbol(i) for i in range(7)]) + " |"



from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])



import argparse
from datetime import datetime
import os
import subprocess
import time
from typing import Any, Dict, Tuple

import ray
from ray import tune
from ray.rllib.agents.ppo import ppo


TRAINER_CFG = {
    "env": MysteriousCorridor,
    "env_config": {
        "reward": 10.0,
    },
    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "model": {
        "custom_model": TorchCustomModel,
        "fcnet_hiddens": [20, 20],
        "vf_share_layers": True,
    },
    "num_workers": 1,  # parallelism
    "framework": "torch",
    "rollout_fragment_length": 10,
    "lr": 0.01,
}

RUN_PREFIX = "CUJ-RL"


def train() -> str:
    print("Training & tuning automatically with Ray Tune...")
    local = True

    run_name = f"{RUN_PREFIX}"
    results = tune.run(
        ppo.PPOTrainer,
        name=run_name,
        config=TRAINER_CFG,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        sync_config=None,
        stop={"training_iteration": 10},
        num_samples=1 if local else 10,
        metric="episode_reward_mean",
        mode="max")

    breakpoint()
    print("Best checkpoint: ")
    print(results.best_checkpoint)

    if upload_dir:
        tmp_file = "/tmp/best_checkpoint.txt"
        with open(tmp_file, "w") as f:
            f.write(results.best_checkpoint)
        best_checkpoint_file = os.path.join(
            upload_dir, run_name, "best_checkpoint.txt")
        print("Saving best checkpoint in: ", best_checkpoint_file)

        if upload_dir.startswith("gs://"):
            subprocess.run(["gsutil", "cp", tmp_file, best_checkpoint_file],
                           check=True)
        elif upload_dir.startswith("s3://"):
            subprocess.run(["aws", "s3", "cp", tmp_file, best_checkpoint_file],
                           check=True)
        else:
            raise ValueError("Unknown upload dir type: ", upload_dir)

    return results.best_checkpoint

train()