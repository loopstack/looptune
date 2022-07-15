# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import pdb
import uuid
import os
import sys
import logging

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.timer import Timer

from compiler2_service.analyzers.dataset_exploration.core import Walker


class GreedyWalker(Walker):
    def __init__(
        self, 
        env: CompilerEnv, 
        dataset_uri: str,
        observation: str, 
        reward: str, 
        walk_count: int, 
        step_count: int,
        max_base_opt: int = 30, # CLANG -O3 has ~150 passes
        seek_count: int = -1 # number of actions you try before deciding
        ):

        Walker.__init__(self, 
                        env=env,
                        dataset_uri=dataset_uri,
                        observation=observation,
                        reward=reward,
                        walk_count=walk_count,
                        step_count=step_count,
                        max_base_opt=max_base_opt)
        
        self.seek_count = min(seek_count, env.action_space.n)
        

    ####################################################################
    # Overwrite functions
    ####################################################################
    def walk(self, step_count: int, baseline_opt: list): 
        # use format_log for appending new enterence in list you return
        self.prev_actions = baseline_opt
        rewards = {}

        for self.step_num in range(1, step_count + 1):
            self.env.send_param("save_state", "0")

            for action_str in random.sample(self.env.action_space.names, self.seek_count):
                action_index = self.env.action_space.from_string(action_str)

                with Timer() as step_time:
                    observation, reward, done, info = self.env.step(
                        action_index, 
                        seek=True,
                        observation_spaces=self.observation,
                        reward_spaces=self.reward
                    )
                    rewards[action_index] = reward[0]

                    if done:
                        logging.critical("Episode ended by environment")
                        break

            best_action_index = max(rewards, key=rewards.get)
            self.env.send_param("save_state", "1")
            self.env.step(best_action_index)                
            self.prev_actions.append(self.env.action_space.names[best_action_index])


