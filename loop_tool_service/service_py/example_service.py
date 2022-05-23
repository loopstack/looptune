#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""An example CompilerGym service in python."""

import logging
import os
import pdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from env import loop_tool_env

import compiler_gym.third_party.llvm as llvm
from compiler_gym import site_data_path
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    Space,
    NamedDiscreteSpace,
    Event,
    ObservationSpace,
    DoubleRange,
    SendSessionParameterReply,
    ByteSequenceSpace,
    BytesSequenceSpace,
    Int64Range,
    CommandlineSpace,
    StringSpace,
    DoubleSequenceSpace,
    DoubleBox,
    DoubleTensor,
    FloatRange
)
from compiler_gym.service.runtime import create_and_run_compiler_gym_service

import utils
import signal
import sys

import loop_tool as lt


class LoopToolCompilationSession(CompilationSession):
    """Represents an instance of an interactive compilation session."""

    compiler_version: str = "1.0.0"

    # keep it simple for now: 1 variable, 1 nest
    action_spaces = [
        ActionSpace(
            name="loop_tool",
            space=Space(
                # potentially define new splits
                named_discrete=NamedDiscreteSpace(
                    name=[
                        "dummy",
                        "up", 
                        "down", 
                        "swap_up", 
                        "swap_down", 
                        "split_2", 
                        "split_4", 
                        "split_8", 
                        "split_16", 
                        "split_32", 
                        "split_64", 
                        # "split_128", 
                        # "split_256", 
                        # "split_512", 
                        # "split_1024", 
                        # "split_2048", 
                        # "split_4096", 
                        # "split_8192", 
                        "merge", 
                        "unroll", 
                        "vectorize", 
                        # "copy_input_0", 
                        # "copy_input_1"
                        ],
                ),
            ),
        ),
    ]

    observation_spaces = [
        ObservationSpace(
            name="runtime_tensor",
            space=Space(
                double_box=DoubleBox(
                    low = DoubleTensor(shape = [1], value=[0]),
                    high = DoubleTensor(shape = [1], value=[float("inf")]),
                )
            ),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
        ObservationSpace(
            name="runtime",
            space=Space(double_value=DoubleRange()),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
        ObservationSpace(
            name="flops",
            space=Space(double_value=DoubleRange()),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
        ObservationSpace(
            name="flops_loop_nest",
            space=Space(double_value=DoubleRange()),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
        ObservationSpace(
            name="loop_tree_ir",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(
                string_value="",
            ),
        ),
    ]
    

    def __init__(
        self,
        working_directory: Path,
        action_space: ActionSpace,
        benchmark: Benchmark,
    ):
        super().__init__(working_directory, action_space, benchmark)
        logging.info(f"Started a compilation session for {benchmark.uri}")
        self._action_space = action_space

        os.chdir(str(working_directory))
        logging.critical(f"\n\nWorking_dir = {str(working_directory)}\n")
        pdb.set_trace()

        self.save_state = False
        
        self.env = loop_tool_env.Environment(
                            working_directory=working_directory,
                            benchmark=benchmark,
                            timeout_sec=3000
        )

        self.prev_observation = {}


    def handle_session_parameter(self, key: str, value: str) -> Optional[str]:
        if key == "save_state":
            self.save_state = False if value == "0" else True
            return "Succeeded"
        else:
            logging.critical("handle_session_parameter Unsuported key:", key)
            return ""


    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        num_choices = len(self.action_spaces[0].space.named_discrete.name)

        # Vladimir: I guess choosing multiple actions at once is not possible anymore.
        # if len(action.int64_value) != 1:
        #     raise ValueError("Invalid choice count")

        choice_index = action.int64_value
        if choice_index < 0 or choice_index >= num_choices:
            raise ValueError("Out-of-range")

        # Compile benchmark with given optimization
        action = self._action_space.space.named_discrete.name[choice_index]
        logging.info(
            f"Applying action {choice_index}, equivalent command-line arguments: '{action}'"
        )

        action_had_effect = self.env.apply_action(action=action, save_state=self.save_state)          

        logging.info(f"\naction_had_effect ({action}) = {action_had_effect}\n")

        if action_had_effect:
            self.prev_observation = {} # Clear cache if action had an effect

        end_of_session = False
        new_action_space = None
        return (end_of_session, new_action_space, not action_had_effect)




    def get_observation(self, observation_space: ObservationSpace) -> Event:
        logging.info(f"Computing observation from space {observation_space.name}")  

        if observation_space.name in self.prev_observation:            
            logging.info(f"get_observation: Fast return prev_observation {self.prev_observation}")
            return self.prev_observation[observation_space.name]

        if observation_space.name == "runtime":
            observation = self.env.get_runtime()
        elif observation_space.name == "flops":
            observation = self.env.get_flops()
        elif observation_space.name == "flops_loop_nest":
            observation = self.env.get_flops_loop_nest()
        elif observation_space.name == "loop_tree_ir":
            observation = self.env.get_ir()
        else:
            raise KeyError(observation_space.name)

        self.prev_observation[observation_space.name] = observation

        logging.info(f"get_observation: Slow return prev_observation {self.prev_observation}")
        return self.prev_observation[observation_space.name]


    def fork(self):
        # There is a problem with forking.
        from copy import deepcopy
        # FIXME vi3: I don't know what is the proper way to fork a session.
        new_fork = deepcopy(self)
        return new_fork


if __name__ == "__main__":
    create_and_run_compiler_gym_service(LoopToolCompilationSession)
