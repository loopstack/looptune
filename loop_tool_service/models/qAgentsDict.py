from networkx.drawing.nx_pydot import to_pydot

from cgi import test
from re import S
from turtle import title
from loop_tool_service.models.qAgentBase import QAgentBase
from matplotlib import pyplot as plt

import random, math

import os
import sys
import pdb
import networkx as nx
import pickle
import json



class QAgentLoopTree(QAgentBase):
    def __init__(self, **args):
        QAgentBase.__init__(self, **args)
        self.observation = "loop_tree"
        self.Q = {}


    def initQ(self, state_hash):
        if state_hash not in self.Q:
            self.Q[state_hash] = {}
            available_actions = self.getAvailableActions(state_hash)
            for a in available_actions:
                self.Q[state_hash][a] = 0

    def hashState(self, state):
        return state

    def getQValues(self, state) -> dict:
        self.initQ(state)
        return self.Q[state]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state.hash)
        self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state.hash, action) )



class QAgentNetworkX(QAgentBase):
    def __init__(self, **args):
        QAgentBase.__init__(self, **args)
        self.Q = {}


    def initQ(self, state_hash):
        if state_hash not in self.Q:
            self.Q[state_hash] = {}
            available_actions = self.getAvailableActions(state_hash)
            for a in available_actions:
                self.Q[state_hash][a] = 0

    def hashState(self, state):
        state_unpickled = pickle.loads(state)
        nx.drawing.nx_pydot.write_dot(state_unpickled,sys.stdout)
        return nx.weisfeiler_lehman_graph_hash(state_unpickled, node_attr='feature_vector')

    def getQValues(self, state) -> dict:
        self.initQ(state.hash)
        return self.Q[state.hash]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state.hash)
        self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state.hash, action) )
