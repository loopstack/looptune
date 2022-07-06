from lib2to3.refactor import get_all_fix_names
from statistics import mean
from tokenize import Double
import numpy as np
import pdb
from loop_tool_service.service_py.utils import run_command_get_stderr
import re
import loop_tool as lt
import networkx as nx
import pydot
import ast

from compiler_gym.service.proto import (
    Event,
    BooleanTensor,
    BooleanRange,
    BooleanBox,
    Int64Tensor,
    FloatTensor,
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
    ByteTensor,
)

# from compiler_gym.util.commands import Popen, run_command
from loop_tool_service.service_py.utils import run_command, proto_buff_container_to_list, print_list, run_command_stdout_redirect



class Environment:
    def __init__(self, 
        working_directory: Path, 
        action_space, 
        observation_spaces,
        benchmark: Benchmark, 
        timeout_sec: float):

        self.name = "loop_tool_env"
        self.action_space = action_space
        self.observation_spaces = { v.name: v.space for v in observation_spaces }
        self.timeout_sec = timeout_sec        

        ir = lt.deserialize(benchmark.program.contents)
        self.agent = lt.LoopTreeAgent(lt.LoopTree(ir))
        logging.info(self.agent)
        self.action_had_effect = False
        self.actions = []


    def get_available_actions(self):
        return self.agent.get_available_actions()

    ##############################################################
    # Apply action
    ##############################################################
    def apply_action(self, action: str, save_state: bool) -> bool:
        agent_copy = lt.LoopTreeAgent(self.agent)
        self.agent.apply_action(action)

        if agent_copy.lt.dump() != self.agent.lt.dump():
            self.action_had_effect = True
        else:
            self.action_had_effect = False

        if save_state == False:
            self.agent = lt.LoopTreeAgent(agent_copy)
        else:
            self.actions.append(action)

        return self.action_had_effect

    ##############################################################
    # Get observations
    ##############################################################
    def get_runtime(self) -> Event:
        mean_runtime = self.agent.eval("seconds")
        return Event(float_value=mean_runtime)

    def get_flops(self) -> Event:
        return Event(float_value=self.agent.eval("FLOPS"))

    def get_flops_loop_nest(self) -> Event:
        with lt.Backend("loop_nest"):
            return Event(float_value=self.agent.eval("FLOPS"))
            
    def get_flops_loop_nest_tensor(self) -> Event:
        with lt.Backend("loop_nest"):
            tensor = DoubleTensor(shape = [1], value=[self.agent.eval("FLOPS")])
        logging.info(f'<<<<<<<<<<<<<<< Reward = {tensor.value[0] / 1e9} GFLOPS >>>>>>>>>>>>>>>')
        return Event(double_tensor=tensor)

    def get_ir(self) -> Event:
        return Event(string_value=self.agent.lt.ir.serialize())

    def get_ir_tree_networkx(self) -> Event:
        pickled = pickle.dumps(self.ir_to_networkx(self.agent.dot_tree()))
        return Event(byte_tensor=ByteTensor(shape=[len(pickled)], value=pickled))

    def get_ir_graph_networkx(self) -> Event:
        pickled = pickle.dumps(self.ir_to_networkx(self.agent.dot_graph()))
        return Event(byte_tensor=ByteTensor(shape=[len(pickled)], value=pickled))

    def get_stride_tensor(self) -> Event:
        dim0, bucket_num = self.observation_spaces['stride_tensor'].float_box.high.shape
        assert(dim0 == 1)
        stride_freq_vector = [0] * bucket_num
        stride_freq_pairs = self.agent.get_stride_frequency()
        total_freq = sum([ x[1] for x in stride_freq_pairs] )
        for stride, freq in stride_freq_pairs:
            bucket_id = int(np.log2(stride))
            stride_freq_vector[bucket_id] += freq/total_freq # Normalize freq

        return Event(float_tensor=FloatTensor(shape=[dim0, bucket_num], value=stride_freq_vector))
    
    def get_loops_tensor(self) -> Event:
        feature_vector = [x for loop_vector in self.agent.get_loops_tensor() for x in loop_vector]
        dim0, dim1 = self.observation_spaces['loops_tensor'].float_box.high.shape
        assert(len(feature_vector) < dim1)
        feature_vector.extend([0] * (dim1 - len(feature_vector)))
        return Event(float_tensor=FloatTensor(shape=[dim0, dim1], value=feature_vector))
    
    def get_loop_tree(self) -> Event:
        return Event(string_value=self.agent.dump())

    def get_prev_actions(self) -> Event:
        dim0, dim1 = self.observation_spaces['5_prev_actions_tensor'].float_box.high.shape
        last_actions_vector = []
        for action in reversed(self.actions[-5:]):
            one_hot_action = [ a == action for a in self.action_space.space.named_discrete.name ]
            last_actions_vector.extend(one_hot_action)

        last_actions_vector.extend([0] * (dim1 - len(last_actions_vector)))

        return Event(float_tensor=FloatTensor(shape=[dim0, dim1], value=last_actions_vector))


    ##############################################################
    # Auxilary functions
    ##############################################################
    def ir_to_networkx(self, dot_str):
        pg = pydot.graph_from_dot_data(str(dot_str))
        gg = nx.nx_pydot.from_pydot(pg[0])


        for nid in list(gg.nodes):
            if nid[0] in ['L', 'D']:
                gg.nodes[nid]["feature_vector"] = self.extract_features(gg.nodes[nid]["feature_dict"])
            else:
                gg.remove_node(nid)
        return gg

    def extract_features(self, label, max_feature_size = 50):
        dict = ast.literal_eval(ast.literal_eval(label))
        return list(dict.values())