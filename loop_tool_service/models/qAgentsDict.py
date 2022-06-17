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
        self.Q = {}


    def initQ(self, state):
        if state not in self.Q:
            self.Q[state] = {}
            available_actions = self.getAvailableActions(state)
            for a in available_actions:
                self.Q[state][a] = 0

    def hashState(self, state):
        return state

    def getQValues(self, state) -> dict:
        self.initQ(state)
        return self.Q[state]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state)
        self.Q[state][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state, action) )



class QAgentNetworkX(QAgentBase):
    def __init__(self, **args):
        QAgentBase.__init__(self, **args)
        self.Q = {}

        self.observation = "ir_networkx"
        self.reward = "flops_loop_nest"

    def initQ(self, state):
        if state not in self.Q:
            self.Q[state] = {}
            available_actions = self.getAvailableActions(state)
            for a in available_actions:
                self.Q[state][a] = 0

    def hashState(self, state):
        state_unpickled = pickle.loads(state)
        # nx.drawing.nx_pydot.write_dot(state_unpickled,sys.stdout)
        return nx.weisfeiler_lehman_graph_hash(state_unpickled, node_attr='feature')

    def getQValues(self, state) -> dict:
        self.initQ(state)
        return self.Q[state]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state)
        self.Q[state][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state, action) )
