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
import json
import random
from matplotlib import pyplot as plt
import pandas as pd
from itertools import cycle
from loop_tool_service.paths import LOOP_TOOL_ROOT

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
        self.agent_saved = None
        self.model = None
        self.device = 'cpu'

    def get_available_actions(self, agent=None):
        def intersection(l1, l2):
            return [ x for x in l1 if x in l2 ]

        if agent == None:
            agent = self.agent
        available_actions = intersection(agent.get_available_actions(), 
                                         self.action_space.space.named_discrete.name)
        return available_actions

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
        flops = self.eval_ln_flops(self.agent)
        return Event(float_value=flops)
            
    def get_flops_loop_nest_tensor(self) -> Event:
        flops = self.eval_ln_flops(self.agent)
        tensor = DoubleTensor(shape = [1], value=[flops])
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
        stride_freq_vector = self.agent.get_stride_histogram()
        assert(dim0 == 1), 'get_stride_tensor:dim0 == 1'
        assert(len(stride_freq_vector) == bucket_num), 'get_stride_tensor:LoopTool dimension doesnt correspond to environment dimensions'
        return Event(float_tensor=FloatTensor(shape=[dim0, bucket_num], value=stride_freq_vector))
    
    def get_loops_tensor(self) -> Event:
        feature_vector = [x for loop_vector in self.agent.get_loops_tensor() for x in loop_vector]
        dim0, dim1 = self.observation_spaces['loops_tensor'].float_box.high.shape
        assert(len(feature_vector) < dim1), f'get_loops_tensor:LoopTool dimension doesnt correspond to environment dimensions {len(feature_vector)} !< {dim1}'
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





    def load_model(self, model_path_str):
        if model_path_str == '':
            self.model = None
        else:
            self.model = torch.jit.load(model_path_str).to(self.device)
            self.model.eval()
            # self.model = SmallNet.load_model(model_path_str)
            # self.model.load_state_dict(torch.load(model_path_str))


    ##############################################################
    # Search functions
    ##############################################################
    def plot_search(self, df, color, linewidth=1):
        
        plt.plot(df['time'], df['measured_reward'], color=color, linewidth=linewidth)
        plt.plot(df['time'], df['predicted_reward'], color=color, linewidth=linewidth)

        # axs[0].scatter(df['time'].iloc[-1], final_reward, c=color, marker='o')

        # axs[1].title.set_text('Test rewards')
        # axs[1].plot(test_rewards, color="green")
        # axs[1].plot(np.zeros_like(test_rewards), color="blue")
        # return axs


    

    def explore_benchmark(self, walk_count, step_count, search_depth, search_width) -> None:
        rewards_actions = []
        cycol = cycle('bgrcmk')
        # fig, axs = plt.subplots(1, 2)
        plt.cla()
        plt.title('Benchmark performance')

        with Timer() as episode_time:
            start_flops = self.eval_ln_flops(self.agent) / 1e9

            for self.walk_num in range(1, walk_count + 1):
                df = self.walk(
                    step_count=step_count, 
                    search_depth=search_depth, 
                    search_width=search_width
                )
                rewards_actions.append(df)
                self.plot_search(df, color = next(cycol))

                # print(f'{start_flops} -> {rewards_actions[-1][0]} GFLOPs, Actions = {rewards_actions[-1][1]}')
        
            best_df = max(rewards_actions, key=lambda x: x['predicted_reward'].iloc[-1]) 
            self.plot_search(best_df, 'red', linewidth=3)
            best_actions = best_df['action'].tolist()
            predicted_reward, measured_reward = best_df[['predicted_reward', 'measured_reward']].iloc[-1]
            print(f"Time = {episode_time}, GFLOPS: {start_flops} -> {measured_reward}, ({predicted_reward}) | Actions = {best_actions}---------")


        plt.tight_layout()
        plt.savefig(str(LOOP_TOOL_ROOT) + "/loop_tool_service/models/tmp.png")

        return measured_reward, best_actions

    def walk(self, step_count: int, search_depth: int, search_width: int)-> list: 
        agent_copy = deepcopy(self.agent)
        cur_reward = 0
        
        df_list = []

        with Timer() as step_time:
            for self.step_num in range(1, step_count + 1):
        
                new_action_str, new_reward = self.get_best_next_action(agent_copy, search_depth, search_width)
                # if new_reward >= cur_reward or True:
                cur_reward = new_reward
                agent_copy.apply_action(new_action_str)
                df_list.append([step_time.time, new_action_str, new_reward, self.eval_ln_flops(agent_copy) / 1e9])

                # else:
                #     break

        return pd.DataFrame(df_list, columns=['time','action', 'predicted_reward', 'measured_reward'])


    # Search
    def get_best_next_action(self, agent_copy, search_depth, search_width):
        if self.model != None:
            eval_fn = self.eval_state_model
        else:
            eval_fn = self.eval_ln_flops


        if search_depth == 0:
            next_action = random.choice(self.get_available_actions(agent=agent_copy))
            new_reward = 0
        else:
            next_action, new_reward =  self.get_best_action_helper(
                agent=agent_copy, 
                search_depth=search_depth, 
                search_width=search_width,
                eval_fn=eval_fn
            )

        return next_action, new_reward
        
    def get_best_action_helper(self, agent, search_depth, search_width, eval_fn):
        best_reward = -1
        best_action = None

        if search_depth == 0:
            # self.env.send_param("undo_action", "")
            return best_action, eval_fn(agent)


        available_actions = self.get_available_actions(agent=agent)
        search_width_real = min(len(available_actions), search_width)
        chosen_actions = random.sample(available_actions, search_width_real)
        # print(chosen_actions)
        for action_str in chosen_actions:

            agent_copy = deepcopy(agent)

            # print(f'search_depth = {search_depth}')
            # print(agent.actions)
            # print(agent)
            next_action, new_reward =  self.get_best_action_helper(agent_copy, search_depth - 1, search_width, eval_fn)

            if new_reward > best_reward:
                best_reward = new_reward 
                best_action = action_str

        # env_fork.send_param("undo_action", "")
        # print(best_action, best_reward)
        return best_action, best_reward


    def eval_ln_flops(self, agent):
        try:
            with lt.Backend("loop_nest"):
                return agent.eval("FLOPS")
        except:
            return 0
            
    def eval_state_model(self, agent):
        state_tensor = [ np.log2(x + 1) for x in agent.get_stride_histogram() ]
        state_tensor = torch.tensor(state_tensor).float().to(self.device)
        pred_flops = self.model(state_tensor)
        return pred_flops.item()