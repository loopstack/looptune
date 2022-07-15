import logging
import random
import pdb
import uuid
import os
import sys

from compiler2_service.agent_py.datasets import (
    hpctoolkit_dataset,
    poj104_dataset,
    poj104_dataset_small,
)

import humanize
from absl import app, flags
from tqdm import tqdm

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.flags.benchmark_from_flags import benchmark_from_flags
from compiler_gym.util.flags.env_from_flags import env_from_flags
from compiler_gym.util.shell_format import emph
from compiler_gym.util.timer import Timer

from joblib import Parallel, delayed


class Walker:
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

        ####################################################################
        # Initialization 
        ####################################################################
        self.env = env
        self.observation = observation.split(',')
        self.reward = reward.split(',')
        self.walk_count = walk_count
        self.step_count = step_count        
        self.max_base_opt = max_base_opt

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
        for bench in tqdm(data_set, total=len(data_set)):
            self.explore_benchmark(bench)

    def explore_benchmark(self, bench: str) -> None:
        """Perform a random walk of the action space.

        :param env: The environment to use.
        :param step_count: The number of steps to run. This value is an upper bound -
            fewer steps will be performed if any of the actions lead the
            environment to end the episode.
        """
        log = []
        rewards = []
            
        with Timer() as episode_time:
            self.bench_uri = str(bench)       

            for self.walk_num in range(1, self.walk_count + 1):
                base_opt_num = random.randrange(self.max_base_opt)
                baseline_opt = random.sample(self.env.action_space.flags, k=base_opt_num)
                self.env.reset(bench)
                self.env.send_param("save_state", "1")

                self.env.multistep(
                    actions=[self.env.action_space.from_string(a) for a in baseline_opt],
                    observation_spaces=self.observation,
                    reward_spaces=self.reward
                    )

                self.walk(self.step_count, baseline_opt)
    

    ####################################################################
    # Overwrite functions
    ####################################################################
    def walk(self, step_count: int, baseline_opt: list)-> list: 
        logging("class Walker: You must implement walk function")
        raise NotImplementedError()


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
             