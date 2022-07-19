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
import json
from copy import deepcopy

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.timer import Timer

from loop_tool_service.models.autotuners.core import Walker


class GreedyWalker(Walker):
    def __init__(
        self, 
        env: CompilerEnv, 
        dataset_uri: str,
        reward: str, 
        walk_count: int, 
        step_count: int,
        seek_count: int = float('inf'), # number of actions you try before deciding
        search_depth: int = 1,
        search_width: int = 10000,
        bench_count: int = 1000000000,
        model_path = None,
        ):

        Walker.__init__(self, 
                        env=env,
                        dataset_uri=dataset_uri,
                        reward=reward,
                        walk_count=walk_count,
                        step_count=step_count,
                        bench_count=bench_count,
                        model_path=model_path)
        
        self.search_depth = search_depth
        self.search_width = search_width
        # self.eval_state_fn = eval_state_fn if eval_state_fn else self.eval_state

    ####################################################################
    # Overwrite functions
    ####################################################################

    def next_action(self):        
        action, reward = json.loads(self.env.send_param('next_best_action', f'{self.search_depth},{self.search_depth}')) 
        return action, reward