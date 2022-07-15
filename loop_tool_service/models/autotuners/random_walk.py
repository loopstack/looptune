# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import pdb
import uuid
import os
import sys

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.timer import Timer
import logging

from compiler2_service.analyzers.dataset_exploration.core import Walker


class RandomWalker(Walker):
    def __init__(
        self, 
        env: CompilerEnv, 
        dataset_uri: str,
        observation: str, 
        reward: str, 
        walk_count: int, 
        step_count: int,
        max_base_opt: int = 30 # CLANG -O3 has ~150 passes
        ):

        Walker.__init__(self, 
                        env=env,
                        dataset_uri=dataset_uri,
                        observation=observation,
                        reward=reward,
                        walk_count=walk_count,
                        step_count=step_count,
                        max_base_opt=max_base_opt)

    ####################################################################
    # Overwrite functions
    ####################################################################
    def walk(self, step_count: int, baseline_opt: list)-> list: 
        # use format_log for appending new enterence in list you return
        self.prev_actions = baseline_opt
        rewards = []

        for self.step_num in range(1, step_count + 1):

            action_index = self.env.action_space.sample()
            action_str = self.env.action_space.names[action_index]

            with Timer() as step_time:
                observation, reward, done, info = self.env.step(
                action_index, 
                observation_spaces=self.observation,
                reward_spaces=self.reward
                )
            

                self.print_log(
                    cur_action=action_str,
                    reward=reward[0],
                    action_had_no_effect=info.get('action_had_no_effect'),
                    time=step_time
                )

                self.prev_actions.append(action_str)
                rewards.append(reward[0])

                if done:
                    logging.info("Episode ended by environment")
                    break

