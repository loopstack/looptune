"""This module defines and registers the example gym environments."""
from pathlib import Path

from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.spaces import Commandline, CommandlineFlag
from compiler_gym.service.proto import Space, CommandlineSpace
from compiler_gym.wrappers import CompilerEnvWrapper
from typing import cast, List, Union, Optional
import os
import shutil
import sys
import logging
import signal
import pickle
import pdb
import pandas as pd
import copy



class LoopToolCompilerEnv(CompilerEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        


class LoopToolCompilerEnvWrapper(CompilerEnvWrapper):
    def __init__(self, env, logging=False):
        super().__init__(env)
        self.logging = logging
        self.log_list = []
        self.prev_observation = None
        # try:
        #     signal.signal(signal.SIGINT, self.log_to_file)
        # except Exception:
        #     print("Problem while registering the CTRL+C event")
        #     # FIXME: See what to do when multiple threads are running within the same process.
        #     import traceback
        #     traceback.print_exc()

    def step(  # pylint: disable=arguments-differ
        self,
        action,
        seek = False,
        observation_spaces = None,
        reward_spaces = None,
        observations = None,
        rewards = None,
    ):
        if observations is not None:
            logging.warn(
                "Argument `observations` of CompilerEnv.step has been "
                "renamed `observation_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            observation_spaces = observations
        if rewards is not None:
            logging.warn(
                "Argument `rewards` of CompilerEnv.step has been renamed "
                "`reward_spaces`. Please update your code",
                category=DeprecationWarning,
            )
            reward_spaces = rewards
        return self.multistep(  actions=[action], 
                                seek=seek, 
                                observation_spaces=observation_spaces, 
                                reward_spaces=reward_spaces)


    def multistep(
        self,
        actions,
        seek=False,
        observation_spaces=None,
        reward_spaces=None,
        **kwargs
    ):
        logging.info("*******  **************** Apply multi-step ***********************")

        if seek: self.env.send_param("save_restore", "0")
        observation, reward, done, info = super().multistep(actions, observation_spaces, reward_spaces, **kwargs)
        if seek: self.env.send_param("save_restore", "1")

        # Log only when you have 1 action 
        if self.logging and len(actions) == 1 and observation:     
            # Log only if you have previous_observation
            if type(self.prev_observation) != type(None) and info.get('action_had_no_effect') == False:
                logging.critical(f"Action = {actions[0]}, No_effect = {info.get('action_had_no_effect')}, reward = {reward[0]}")
                self.log_list.append(
                    self.format_log(
                        benchmark_uri=str(self.env.benchmark.uri),
                        observation_names = observation_spaces,
                        prev_observation=self.prev_observation,
                        observation=observation,                        
                        action=self.env.action_space.names[actions[0]],
                        prev_actions=self.env.commandline(),
                        reward=reward[0] if not isinstance(reward, float) else reward,                        
                    )
                )
            self.prev_observation = copy.deepcopy(observation)

        # if seek:
        #     self.env.actions = self.env.actions[:-len(actions)]

        return observation, reward, done, info

    # def observation(self, observation):
    #     return np.concatenate((observation, self.histogram)).astype(
    #         self.env.observation_space.dtype
    #     )

    def close(self):
        # Dump current content of the log_list to a file.
        # self.log_to_file()
        super().close()

    # def __del__(self):
    #     # In case someone forgot to call close for the env.
    #     self.log_to_file()

    @staticmethod
    def create_log_dir(env_name):
        from datetime import datetime

        root = os.getenv('LOOP_TOOL_ROOT')
        assert root
        timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        log_dir = "/".join([root, "results", "random-" + env_name, timestamp, str(os.getpid())])
        os.makedirs(log_dir)

        # Put executed command to the log 
        with open(log_dir + "/command.txt", "w") as txt:
            txt.write(" ".join(sys.argv))


        return log_dir

    @staticmethod
    def list2str(l):
        return " ".join([str(x) for x in l])

    @staticmethod
    def format_log(
        benchmark_uri: str, 
        observation_names: list,
        prev_observation: list, 
        observation: list, 
        action: str, 
        prev_actions: str, 
        reward: float,        
        ):
        
        prev_obs_ret = []
        obs_ret = []
        for i, obs_name in enumerate(observation_names):
            if obs_name.endswith("tensor"):
                prev_obs_ret.append(prev_observation[i].flat[:])
                obs_ret.append(observation[i].flat[:])
            elif obs_name.endswith("pickle"):
                prev_obs_ret.append(pickle.loads(prev_observation[i]))
                obs_ret.append(pickle.loads(observation[i]))

            else:
                logging.critical(f"FormatLog doesn't recognize Observation Type: {obs_name}")
                exit(1)

        return [benchmark_uri,
                prev_obs_ret,
                obs_ret,
                action,
                prev_actions,
                reward]


    # def log_to_file(self):
    #     if len(self.log_list) == 0:
    #         return

    #     log_path = self.create_log_dir(self.env.spec.id)

    #     columns = ["BenchmarkName", "State", "NextState", "Action", "CommandLine", "Reward"]
    #     df = pd.DataFrame(self.log_list, columns=columns) 
    #     df.head()
    #     with open(log_path + '/results.pkl', 'wb') as f:
    #         pickle.dump(df, f)
        
    #     print("\nResults written to: ", log_path + "/results.pkl\n")
    #     # Clear the content
    #     self.log_list.clear()





from compiler_gym.util.registration import register
from loop_tool_service.paths import LOOP_TOOL_SERVICE_PY
from loop_tool_service.service_py.rewards import runtime_reward
from compiler_gym.envs.llvm.datasets import (
    AnghaBenchDataset,
    BlasDataset,
    CBenchDataset,
    CBenchLegacyDataset,
    CBenchLegacyDataset2,
    CHStoneDataset,
    CsmithDataset,
    NPBDataset,
)


from compiler_gym.util.runfiles_path import site_data_path

import loop_tool_service
from loop_tool_service.service_py.rewards import flops_loop_nest_reward
import importlib


def register_env(datasets, obs='flops_loop_nest_tensor'):
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        # entry_point=LoopToolCompilerEnv,
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.NormRewardTensor(obs=obs), #NormRewardTensor
                ],
            "datasets": [ 
                importlib.import_module(f"loop_tool_service.service_py.datasets.{dataset}").Dataset() for dataset in datasets 
                ],
        },
    )



def make(id: str, datasets, **kwargs):
    """Equivalent to :code:`compiler_gym.make()`."""
    if len(datasets):
        register_env(datasets=datasets, obs=kwargs['reward_space'])

    import compiler_gym
    return compiler_gym.make(id, **kwargs)


def make_env(id, datasets=[], logging=False, **kwargs):
    return LoopToolCompilerEnvWrapper(make(id, datasets, **kwargs), logging=logging)


