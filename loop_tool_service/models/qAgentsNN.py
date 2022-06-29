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


class Q_net(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, dropout):
        super(Q_net,self).__init__()
        self.l1 = nn.Linear(in_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,hidden_size)
        self.l4 = nn.Linear(hidden_size,hidden_size)
        self.l5 = nn.Linear(hidden_size,hidden_size)
        self.l6 = nn.Linear(hidden_size,hidden_size)
        self.l7 = nn.Linear(hidden_size,out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = F.leaky_relu(self.l1(x))
        x = self.dropout(x)
        y = F.leaky_relu(self.l2(x))
        y = self.dropout(y)
        x = (x+y)*0.5
        x = F.leaky_relu(self.l3(x))
        x = self.dropout(x)
        y = F.leaky_relu(self.l4(x))
        y = self.dropout(y)
        x = (x+y)*0.5
        x = F.leaky_relu(self.l5(x))
        x = self.dropout(x)
        y = F.leaky_relu(self.l6(x))
        y = self.dropout(y)
        x = (x+y)*0.5
        return self.l7(x)

class QAgentTensor(QAgentBase):
    def __init__(self, device='cpu', **args):
        QAgentBase.__init__(self, **args)
        self.criterion = nn.SmoothL1Loss()

        observation_space = args['observation']
        size_in = np.prod(args['env'].observation.spaces[observation_space].space.shape)
        size_out = args['env'].action_spaces[0].n

        self.policy_net = Q_net(in_size=size_in, 
                                out_size=size_out, 
                                hidden_size=2 * size_in, 
                                dropout=0.5).to(device)

        self.target_net = Q_net(in_size=size_in, 
                                out_size=size_out, 
                                hidden_size=2 * size_in, 
                                dropout=0.5).to(device)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())


    def hashState(self, state):
        return state.tostring()


    def getQValues(self, state) -> dict:        
        # breakpoint()
        tensor_in = torch.from_numpy(state.state).float()
        tensor_out = self.policy_net.forward(tensor_in)[0]
        q_dict = { x:tensor_out[x] for x in self.getAvailableActions(state.hash)} 
        return q_dict
        

    def update(self, state, action, nextState, reward):
        state_tensor = torch.from_numpy(state.state).float()
        nextState_tensor = torch.from_numpy(nextState.state).float()

        r = torch.sum(torch.mul(self.policy_net(state_tensor), action), 1)
        r_hat = reward + torch.max(self.target_net(nextState_tensor),1).values * self.discount

        loss = self.criterion(r,r_hat)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()