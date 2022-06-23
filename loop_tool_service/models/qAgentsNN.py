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

import torch
import torch.nn as nn

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

        self.policy_net = Q_net(in_size=100, 
                                out_size=4, 
                                hidden_size=512, 
                                dropout=0.5).to(device)

        self.target_net = Q_net(in_size=100, 
                                out_size=4, 
                                hidden_size=512, 
                                dropout=0.5).to(device)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())


    def hashState(self, state):
        return state.tostring()


    def getQValues(self, state) -> dict:
        q_dict = {k: 0 for k in self.getAvailableActions(state.hash)}
        
        q_values = self.q_net.forward(state.fromstring())

        return q_dict
        

    def update(self, state, action, nextState, reward):
        
        r = torch.sum(torch.mul(self.policy_net(state), action), 1)
        r_hat = 1 + torch.max(self.target_net(nextState),1).values * self.discount

        loss = criterion(r,r_hat)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()