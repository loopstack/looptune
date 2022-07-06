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

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Q_net(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout):
        super(Q_net,self).__init__()
        self.l1 = nn.Linear(in_size,hidden_size)
        self.l7 = nn.Linear(hidden_size,out_size)
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x1 = F.leaky_relu(self.l1(x))
        x2 = F.leaky_relu(self.l7(x1))
        return x2
        
class CostAgent(QAgentBase):
    def __init__(self, device='cpu', **args):
        QAgentBase.__init__(self, **args)
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.HuberLoss()

        observation_space = args['observation']
        self.size_in = np.prod(args['env'].observation.spaces[observation_space].space.shape)
        self.size_out = 1

        self.policy_net = Q_net(in_size=self.size_in, 
                                out_size=self.size_out, 
                                hidden_size=self.size_in, 
                                dropout=0.5).to(device)

        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.1)
        # self.optimizer = torch.optim.RMSProp(self.policy_net.parameters(), lr=0.1)
        self.history = []

    def hashState(self, state):
        return state.tostring()


    def getQValues(self, state) -> dict:
        breakpoint()        
        tensor_in = torch.from_numpy(state.state).float()
        
        # Beam search
        q_dict = {}
        # for action in self.getAvailableActions(state.hash):
        #     with self.env.fork() as fork_env:
        #         observation, reward, done, info = fork_env.step(
        #         action=action,
        #         observation_spaces=[self.observation],
        #         reward_spaces=[self.reward],
        #     )
        #     q_dict[action] = reward[0]

        # breakpoint()
        # self.policy_net.forward(tensor_in)[0]

        # breakpoint()
        q_dict = { x:0 for x in self.getAvailableActions(state.hash)} 
        return q_dict
        

    def update(self, state, action, nextState, reward):
        state_tensor = torch.from_numpy(state.state).float()
        # action_onehot = [0] * self.size_out
        # action_onehot[action] = 1
        # action_onehot = torch.tensor(action_onehot)

        breakpoint()
        pred = self.policy_net(state_tensor)[0][0]
        target = torch.tensor(reward)
        loss = self.criterion(pred, target)
        self.loss_history.append(loss.detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print(f'Predicted = {pred} GFLOPS')
        # breakpoint()
        # if action in [1, 3]: breakpoint()
        print('_________________________________________')


