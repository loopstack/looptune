import logging
import random
import pdb
import uuid
import os
import sys

import json
import humanize
from absl import app, flags
from tqdm import tqdm

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.timer import Timer

from joblib import Parallel, delayed


class Walker:
    def __init__(
        self, 
        env: CompilerEnv, 
        dataset_uri: str,
        reward: str, 
        walk_count: int, 
        step_count: int,
        bench_count: int,
        model_path: str,
        ):

        ####################################################################
        # Initialization 
        ####################################################################
        self.env = env
        self.reward = reward
        self.walk_count = walk_count
        self.step_count = step_count        
        self.bench_count = bench_count
        self.model_path = model_path

        # Internal parameters
        self.dataset_uri = dataset_uri
        self.bench_uri = ""
        self.walk_num = 0
        self.step_num = 0
        self.prev_actions = []
        self.log_list = []


    ####################################################################
    # API Function 
    ####################################################################
    def run(self):
        data_set = self.env.datasets[self.dataset_uri]
        i = 0
        for bench in tqdm(data_set, total=min(len(data_set), self.bench_count)):
            if i == self.bench_count: break
            self.explore_benchmark(bench)
            i += 1

    def explore_benchmark(self, bench: str) -> None:
        """Perform a random walk of the action space.

        :param env: The environment to use.
        :param step_count: The number of steps to run. This value is an upper bound -
            fewer steps will be performed if any of the actions lead the
            environment to end the episode.
        """
        log = []
        rewards_actions = []
            
        with Timer() as episode_time:
            self.bench_uri = str(bench)       
            print(self.bench_uri)
            self.env.reset(bench)
        
            start_flops = self.env.observation[self.reward] / 1e9

            for self.walk_num in range(1, self.walk_count + 1):
                self.env.reset(bench)
                if self.model_path != None:
                    self.env.send_param('load_model', self.model_path)

                rewards_actions.append(self.walk(self.step_count))
                print(f'{start_flops} -> {rewards_actions[-1][0]} GFLOPs, Actions = {rewards_actions[-1][1]}')
            
            print(f"--------BEST = {max(rewards_actions)}---------")


    ####################################################################
    # Overwrite functions
    ####################################################################
    def next_action(self):
        print('Implement next_action given environment')
        raise NotImplementedError


    def walk(self, step_count: int)-> list: 
        # use format_log for appending new enterence in list you return
        self.prev_actions = []
        cur_reward = 0

        for self.step_num in range(1, step_count + 1):
        
            new_action_str, new_reward = self.next_action()
            # if new_reward >= cur_reward or True:

            cur_reward = new_reward
            action_index = self.env.action_space.from_string(new_action_str)
            self.env.step(action_index)
            self.prev_actions.append(new_action_str)
            # else:
            #     break
        return self.env.observation[self.reward] / 1e9, self.prev_actions
               



    ####################################################################
    # Auxilary functions
    ####################################################################

    def print_log(self, cur_action, reward, action_had_no_effect, time=""):
        (                
                f"\n********** Bench {self.bench_uri} **********\n"
                f"\n\t=== Walk {humanize.intcomma(self.walk_num)} ===\n"                       
                f"\n\t\t>>> Step {humanize.intcomma(self.step_num)} >>> {time} \n"
                f"Prev Actions: %s\n"%",".join(self.prev_actions),
                f"Action:       {cur_action} "
                f"(changed={not action_had_no_effect})\n"
                f"Reward:       {reward}"
            )
             