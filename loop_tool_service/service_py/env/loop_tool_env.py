from statistics import mean
from tokenize import Double
import numpy as np
import pdb
from loop_tool_service.service_py.utils import run_command_get_stderr
import re
import loop_tool as lt

from compiler_gym.service.proto import (
    Event,
    DoubleTensor,
    DoubleRange,
    DoubleBox,
    DoubleSequenceSpace)

from ctypes import util
import logging
import os
import pdb
import subprocess
from copy import deepcopy as deepcopy
from pathlib import Path
from signal import Signals
from typing import List, Optional, Tuple
from xmlrpc.client import Boolean

import loop_tool_service.paths
import pickle
import loop_tool as lt

import compiler_gym.third_party.llvm as llvm
from compiler_gym.service.proto import Benchmark

from compiler_gym.service.proto import (
    Event,
)

# from compiler_gym.util.commands import Popen, run_command
from loop_tool_service.service_py.utils import run_command, proto_buff_container_to_list, print_list, run_command_stdout_redirect



class Environment:
    def __init__(self, working_directory: Path, benchmark: Benchmark, timeout_sec: float):
        self.name = "loop_tool_env"
        self.timeout_sec = timeout_sec        

        self.tensor = lt.Tensor()
        self.tensor.set(lt.deserialize(benchmark.program.contents))

        self.cursor = 0
        self.action_had_effect = False
        print(self.tensor.loop_tree)



    def update_tree(self, new_tree):
        self.cursor = new_tree.map_ref(self.cursor, self.tensor.loop_tree)
        self.action_had_effect = True
        return new_tree

    # Apply action
    def apply_action(self, action: str, save_state: bool) -> bool:
        # opt format "-opt1 -opt2 ..."

        self.action_had_effect = False
        tree_before = self.tensor.loop_tree    
        tree_after = None

        # pdb.set_trace()

        # Apply action here
        if (action == 'up'):
            p = tree_before.previous_ref(self.cursor)
            if p is not None:
                self.cursor = p
            
        elif(action == "down"):
            n = tree_before.next_ref(self.cursor)
            if n is not None:
                self.cursor = n
        
        elif(action == "swap_up"):
            tree_after = self.update_tree(tree_before.try_swap(self.cursor, tree_before.previous_ref(self.cursor)))

        elif(action == "swap_down"):
            tree_after = self.update_tree(tree_before.try_swap(self.cursor, tree_before.next_ref(self.cursor)))


        elif(action.startswith("split")):
            split_param = int(action.split('_')[-1])
            try:
                tree_after = self.update_tree(tree_before.split(self.cursor, split_param))
            except:
                pass

        elif(action == "merge_up"):
            tree_after = self.update_tree(tree_before.merge(self.cursor))

        elif(action == "vectorize"):
            tree_after = self.update_tree(tree_before.annotate(self.cursor, "vectorize"))

        elif(action == "unroll"):
            tree_after = self.update_tree(tree_before.annotate(self.cursor, "unroll"))


        elif(action == "copy_input_0"):
            tree_after = self.update_tree(tree_before.copy_input(self.cursor, 0))

        elif(action == "copy_input_1"):
            tree_after = self.update_tree(tree_before.copy_input(self.cursor, 1))



        # Check if action made an effect
        if self.action_had_effect:
            self.tensor.set(tree_after)

        return self.action_had_effect

    # Get observations
    def get_runtime(self) -> Event:
        mean_runtime = self.tensor.loop_tree.eval()
        return Event(float_value=mean_runtime)

    def get_flops(self) -> Event:
        mean_runtime = self.tensor.loop_tree.eval()
        flos = self.tensor.loop_tree.flops() 
        return Event(float_value= flos / mean_runtime)

    def get_flops_loop_nest(self) -> Event:
        with lt.Backend("loop_nest"):
            mean_runtime = self.tensor.loop_tree.eval()
            flos = self.tensor.loop_tree.flops()
        return Event(float_value=flos / mean_runtime)


    def get_ir(self) -> Event:
        return Event(string_value=self.tensor.ir.serialize())
 
