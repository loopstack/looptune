from networkx.drawing.nx_pydot import to_pydot

from cgi import test
from re import S
from turtle import title
from loop_tool_service.models import StateAgent
from matplotlib import pyplot as plt

import random, math

import os
import sys
import pdb
import networkx as nx
import pickle
import json


class QAgentLoopTree(StateAgent):
    def __init__(self, **args):
        StateAgent.__init__(self, **args)
        self.observation = "loop_tree"


    # def initQ(self, state_hash):
    #     if state_hash not in self.Q:
    #         self.Q[state_hash] = {}
    #         available_actions = self.getAvailableActions(state_hash)
    #         for a in available_actions:
    #             self.Q[state_hash][a] = 0

    def hashState(self, state):
        return state
