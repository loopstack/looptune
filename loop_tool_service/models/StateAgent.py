from itertools import combinations, permutations
import random,time
import pdb
import json
from matplotlib import pyplot as plt
import numpy as np
from itertools import combinations, permutations


import loop_tool as lt
import networkx as nx

class State:
    def __init__(self, state, state_hash):
        self.state = state
        self.hash = state_hash



class StateAgent():
 
    def __init__(self, 
            env, 
            bench,
            observation,
            reward, 
        ):

        self.env = env
        self.bench = bench
        self.observation = observation
        self.reward = reward
        
        self.Q_counts = {}
        self.state_graph = nx.Graph()


    def hashState(self, state):
        return state


    def create_state_graph(self, depth):
        self.env.reset(benchmark=self.bench)
        self.env.send_param("save_state", "0")

        for d in range(depth):
            for comb in combinations(self.env.action_space.names, d):
                for perm in permutations(comb):
                    print(perm)

                    observation, rewards, done, info = self.env.multistep(
                        actions=[ self.env.action_space.from_string(a) for a in perm],
                        observation_spaces=[self.observation],
                        reward_spaces=[self.reward],
                    )

                    # Create node
                    key = ",".join(perm)
                    prev_key = ",".join(perm[:-1])

                    if key not in self.state_graph:
                        self.state_graph.add_node(key)
                        self.state_graph.nodes[key]["V"] = rewards[0]
                        
                        # Create edges
                        self.state_graph.add_edge(prev_key, key)
                    


    def find_best_action_sequence():
        # Graph Search
        pass

    def optimize_policy(self):
        # Policy iteration
        pass


import logging
from compiler_gym.util.registration import register
from compiler_gym.util.logging import init_logging
import loop_tool_service


from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import runtime_reward, flops_reward


def register_env():
    register(
        id="loop_tool-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_reward.RewardScalar(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
            ],
        },
    )

def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    bench = "benchmark://loop_tool_simple-v0/mm128"

    with loop_tool_service.make_env("loop_tool-v0") as env:
        agent = StateAgent(
            env=env,
            bench=bench,
            observation = "loop_tree",
            reward="flops",
            )
        pdb.set_trace()
        agent.create_state_graph(depth=4)
        pdb.set_trace()
        agent.find_best_action_sequence()
        agent.optimize_policy()


if __name__ == "__main__":
    main()  