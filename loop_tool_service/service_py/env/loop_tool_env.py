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
 
from loop_tool_service.service_py.env.evaluator import Evaluator
from loop_tool_service.service_py.env.search import *


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
        self.agent = lt.LoopTreeAgent(lt.LoopTree(ir))#.merge_all()
        self.agent.set_action_space(self.action_space_str)
        logging.info(self.agent)
        self.lt_changed = False
        self.agent_saved = None

       
        self.evaluator = Evaluator(self)
        self.beam_searcher = BeamSearcher(self.evaluator)
        self.greedy_searcher = GreedySearcher(self.evaluator)
        self.policy_cost_searcher = PolicyCostSearcher(self.evaluator)

        self.cost_model = None
        self.policy_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_cost_fn = self.eval_ln_flops


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
        mean_runtime = self.agent.eval("seconds")
        return Event(float_value=mean_runtime)

    def get_flops(self) -> Event:
        return Event(float_value=self.agent.eval("FLOPS") / 1e9)

    def get_flops_loop_nest(self) -> Event:
        flops = self.eval_ln_flops(self.agent)
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
        assert(len(feature_vector) < dim1), f'get_loops_tensor:LoopTool dimension doesnt correspond to environment dimensions {len(feature_vector)} !< {dim1}'
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



    ##############################################################
    # Search functions
    ##############################################################
    '''
        Format search(agent) -> list(actions), list(rewards
    '''
    def policy_beam_search(self, search_depth=10, num_strategies=1):
        actions_reward_pairs = []
        actions_reward_pairs_policy = self.policy_search_get_all(search_depth, num_strategies)
        agent = self.agent.copy()

        for actions in actions_reward_pairs_policy:
            for action in actions:
                agent_copy = agent.copy()
                agent_copy.apply_action(action)
                actions_reward_pairs.append(self.greedy_search(agent_copy))
                

        return actions_reward_pairs



    def policy_search_get_all(self, search_depth=10, num_strategies=1):
        if self.policy_model == None:
            print('Instantiate policy model with: env.send_param("load_policy_model", policy_model_path)')
            return ["", 0]

        graph = nx.DiGraph()
        actions_reward_pairs = []
        self.get_best_actions_helper(self.agent, actions_reward_pairs, search_depth=search_depth, num_strategies=num_strategies, graph=graph)
        print(nx.nx_pydot.to_pydot(graph))
        print(actions_reward_pairs)
        return actions_reward_pairs


    def policy_search(self, search_depth=10, num_strategies=1):
        actions_reward_pairs = self.policy_search_get_all(search_depth, num_strategies)

        if len(actions_reward_pairs):
            return max(actions_reward_pairs, key=lambda x: x[1]) 
        else:
            return ["", 0]


    def get_best_actions_helper(self, agent, actions_reward_pairs, search_depth=10, num_strategies=1, graph=nx.DiGraph()):
        if search_depth == 0:
            return
            
        sorted_actions_q, sorted_actions = self.eval_policy_model(agent)
        available_actions_str = agent.get_available_actions()

        avail_sorted_aq = []
        for a, q in zip(sorted_actions.squeeze(), sorted_actions_q.squeeze()):
            a_str = self.action_space_str[a.item()]
            if a_str in available_actions_str:
                avail_sorted_aq.append((a_str, q.item()))


        print(avail_sorted_aq)
        node_key = hash(agent.dump())
        real_flops = self.eval_ln_flops(agent)
        predicted_flops = self.eval_cost_model(agent) if self.cost_model else -1
        graph.add_node(
            node_key, 
            label=f'FLOPS = {real_flops:9.4f}\nPRED = {predicted_flops:9.4f}\n' + agent.dump().replace(':', ';')
            )

        if all([x[1] < 0 for x in avail_sorted_aq]):
            graph.nodes[node_key]['color'] = 'lightblue1'
            graph.nodes[node_key]['style'] = 'filled'
            actions_reward_pairs.append((agent.actions, self.eval_cost_fn(agent)))
            return


        # print(chosen_actions)
        for action_str, action_q in avail_sorted_aq:
            if len(actions_reward_pairs) == num_strategies: return

            agent_copy = agent.copy()
            agent_copy.apply_action(action_str)
            
            loop_detetcted = hash(agent_copy.dump()) in graph
            graph.add_edge(hash(agent.dump()), hash(agent_copy.dump()), label=f'{action_str}\n{action_q:9.4f}', color='black')
            if loop_detetcted: 
                continue

            self.get_best_actions_helper(agent_copy, actions_reward_pairs, search_depth-1, num_strategies, graph)                



    def greedy_search(self, walk_count, num_steps, search_depth, search_width):
        return self.greedy_searcher.search_n(
            agent=self.agent,
            n=walk_count, 
            num_steps=num_steps,
            lookahead=search_depth, 
            search_width=search_width,
            eval_mode="policy",
        )

    def policy_cost_search(self, num_steps, search_width, n):
        return self.policy_cost_searcher.search(
            agent=self.agent,
            num_steps=num_steps,
            search_width=search_width,
            n=n, 
        )

    