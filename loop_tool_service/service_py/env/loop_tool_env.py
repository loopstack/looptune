from lib2to3.refactor import get_all_fix_names
from statistics import mean
from sys import float_repr_style
from tokenize import Double
import numpy as np
import pdb
from loop_tool_service.service_py.utils import run_command_get_stderr
import re
import loop_tool as lt
import networkx as nx
import pydot
import ast
import json
import random
from matplotlib import pyplot as plt
import pandas as pd
from itertools import cycle
from loop_tool_service.paths import LOOP_TOOL_ROOT
 
from loop_tool_service.service_py.env.evaluator_env import Evaluator
from loop_tool_service.service_py.env.search_env import *
import time

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
from compiler_gym.util.timer import Timer

from compiler_gym.service.proto import (
    Event,
    ByteTensor,
)

# from compiler_gym.util.commands import Popen, run_command
from loop_tool_service.service_py.utils import run_command, proto_buff_container_to_list, print_list, run_command_stdout_redirect

import torch 
import networkx as nx


class Environment:
    def __init__(self, 
        working_directory: Path, 
        action_space, 
        observation_spaces,
        benchmark: Benchmark, 
        timeout_sec: float):

        self.name = "loop_tool_env"
        self.action_space_str = action_space.space.named_discrete.name
        self.observation_spaces = { v.name: v.space for v in observation_spaces }
        self.timeout_sec = timeout_sec        

        ir = lt.deserialize(benchmark.program.contents)
        self.agent = lt.LoopTreeAgent(lt.LoopTree(ir)).merge_all()
        self.agent.set_action_space(self.action_space_str)
        self.agent_start = self.agent.copy()
        logging.info(self.agent)
        self.lt_changed = False
        self.agent_saved = None

        self.cost_model = None
        self.policy_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_cost_fn = self.eval_ln_flops
        self.cache = {}

        self.evaluator = Evaluator(self)
        self.beam_searcher_dfs = BeamSearcherDFS(self.evaluator)
        self.beam_searcher_bfs = BeamSearcherBFS(self.evaluator)
        self.greedy_searcher = GreedySearcher(self.evaluator)
        self.random_searcher = RandomSearcher(self.evaluator)


    def reset_agent(self):
        self.agent = self.agent_start.copy()
        # self.cache = {} # this make rewards unstable
    ##############################################################
    # Apply action
    ##############################################################
    def apply_action(self, action: str, save_state: bool) -> bool:
        agent_copy = lt.LoopTreeAgent(self.agent)
        self.agent.apply_action(action)
        if agent_copy.lt.dump() != self.agent.lt.dump():
            self.lt_changed = True
            action_had_effect = True
        else:
            self.lt_changed = False
            if agent_copy.cursor != self.agent.cursor:
                action_had_effect = True
            else:
                action_had_effect = False

        if save_state == False:
            self.agent = lt.LoopTreeAgent(agent_copy)

        return action_had_effect

    ##############################################################
    # Get observations
    ##############################################################
    def get_runtime(self) -> Event:
        with lt.Backend("loop_nest"):
            mean_runtime = self.agent.eval("seconds")
        return Event(float_value=mean_runtime)

    def get_flops(self) -> Event:
        return Event(float_value=self.agent.eval("FLOPS") / 1e9)

    def get_flops_loop_nest(self) -> Event:
        flops = self.eval_ln_flops(self.agent)
        return Event(float_value=flops)

    def get_flops_loop_nest_cached(self) -> Event:
        flops = self.eval_ln_flops_cached(self.agent)
        return Event(float_value=flops)

    def get_gflops_cost(self) -> Event:
        gflops_cost = self.evaluator.eval_gflops(self.agent, 'cost')
        return Event(float_value=gflops_cost)

    def get_q_policy(self) -> Event:
        q_policy = self.evaluator.get_actions_q_policy_tensor(self.agent)
        return Event(float_tensor=FloatTensor(shape=[1, len(self.agent.action_space)], value=q_policy))


    def get_flops_loop_nest_tensor(self) -> Event:
        flops = self.eval_ln_flops(self.agent)
        tensor = DoubleTensor(shape = [1], value=[flops])
        logging.info(f'<<<<<<<<<<<<<<< Reward = {tensor.value[0]} GFLOPS >>>>>>>>>>>>>>>')
        return Event(double_tensor=tensor)

    def get_flops_loop_nest_tensor_cached(self) -> Event:
        flops = self.eval_ln_flops_cached(self.agent)
        tensor = DoubleTensor(shape = [1], value=[flops])
        logging.info(f'<<<<<<<<<<<<<<< Reward = {tensor.value[0]} GFLOPS >>>>>>>>>>>>>>>')
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
        stride_freq_vector = self.agent.get_stride_histogram()
        assert(dim0 == 1), 'get_stride_tensor:dim0 == 1'
        assert(len(stride_freq_vector) == bucket_num), 'get_stride_tensor:LoopTool dimension doesnt correspond to environment dimensions'
        return Event(float_tensor=FloatTensor(shape=[dim0, bucket_num], value=stride_freq_vector))
    
    def get_loops_tensor(self, agent=None) -> Event:
        if agent == None: agent = self.agent
        feature_vector = [x for loop_vector in agent.get_loops_tensor() for x in loop_vector]
        dim0, dim1 = self.observation_spaces['loops_tensor'].float_box.high.shape
        assert(len(feature_vector) <= dim1), f'get_loops_tensor:LoopTool dimension doesnt correspond to environment dimensions {len(feature_vector)} !< {dim1}'
        feature_vector.extend([0] * (dim1 - len(feature_vector)))
        return Event(float_tensor=FloatTensor(shape=[dim0, dim1], value=feature_vector))
    
    def get_loop_tree(self) -> Event:
        return Event(string_value=self.agent.dump())

    def get_prev_actions(self) -> Event:
        dim0, dim1 = self.observation_spaces['5_prev_actions_tensor'].float_box.high.shape
        last_actions_vector = []
        for action in reversed(self.agent.actions[-5:]):
            one_hot_action = [ a == action for a in self.action_space_str ]
            last_actions_vector.extend(one_hot_action)

        last_actions_vector.extend([0] * (dim1 - len(last_actions_vector)))

        return Event(float_tensor=FloatTensor(shape=[dim0, dim1], value=last_actions_vector))

    def eval_ln_flops_cached(self, agent=None):
        if agent == None: agent = self.agent
        lt_hash = hash(str(agent.lt))
        if lt_hash not in self.cache:
            self.cache[lt_hash] = max([ self.eval_ln_flops(agent) for _ in range(10) ])
        return self.cache[lt_hash]


    def eval_ln_flops(self, agent):
        try:
            with lt.Backend("loop_nest"):
                return agent.eval("FLOPS") / 1e9
        except:
            return 0

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

