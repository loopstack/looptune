from functools import total_ordering
from networkx.drawing.nx_pydot import to_pydot

from cgi import test
from re import S
from turtle import title
from loop_tool_service.models.q_agents.qAgentBase import QAgentBase
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
        # self.l2 = nn.Linear(hidden_size,hidden_size)
        # self.l3 = nn.Linear(hidden_size,hidden_size)
        # self.l4 = nn.Linear(hidden_size,hidden_size)
        # self.l5 = nn.Linear(hidden_size,hidden_size)
        # self.l6 = nn.Linear(hidden_size,hidden_size)
        self.l7 = nn.Linear(hidden_size,out_size)
        # self.softmax = nn.Softmax(dim=1)
        # self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x1 = F.leaky_relu(self.l1(x))
        x2 = F.leaky_relu(self.l7(x1))
        return x2
        # y = self.dropout(y)
        # x = (x+y)*0.5
        # x = F.leaky_relu(self.l3(x))
        # x = self.dropout(x)
        # y = F.leaky_relu(self.l4(x))
        # y = self.dropout(y)
        # x = (x+y)*0.5
        # x = F.leaky_relu(self.l5(x))
        # x = self.dropout(x)
        # y = F.leaky_relu(self.l6(x))
        # y = self.dropout(y)
        # x = (x+y)*0.5
        # return self.l7(x)

class QAgentTensor(QAgentBase):
    def __init__(self, device='cpu', **args):
        QAgentBase.__init__(self, **args)
        # self.criterion = nn.SmoothL1Loss()
        self.criterion = nn.HuberLoss()

        observation_space = args['observation']
        self.size_in = np.prod(args['env'].observation.spaces[observation_space].space.shape)
        self.size_out = args['env'].action_spaces[0].n

        self.policy_net = Q_net(in_size=self.size_in, 
                                out_size=self.size_out, 
                                hidden_size=self.size_in, 
                                dropout=0.5).to(device)

        self.target_iter = 0
        self.target_net = Q_net(in_size=self.size_in, 
                                out_size=self.size_out, 
                                hidden_size=self.size_in, 
                                dropout=0.5).to(device)

        self.confidence = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.RMSProp(self.policy_net.parameters(), lr=0.1)
        # breakpoint()
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, total_iters=self.numTraining)

    def hashState(self, state):
        return state.tostring()


    def getQValues(self, state) -> dict:        
        tensor_in = torch.from_numpy(state.state).float()
        tensor_out = self.policy_net.forward(tensor_in)[0]
        # breakpoint()
        q_dict = { x:tensor_out[x] for x in self.getAvailableActions(state.hash)} 
        return q_dict
        

    # def update(self, state, action, nextState, reward):
    #     state_tensor = torch.from_numpy(state.state).float()
    #     nextState_tensor = torch.from_numpy(nextState.state).float()

    #     r = torch.sum(torch.mul(self.policy_net(state_tensor), action), 1)
    #     r_hat = reward + torch.max(self.target_net(nextState_tensor),1).values * self.discount

    #     loss = self.criterion(r,r_hat)
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # for name, param in self.policy_net.named_parameters(): print(name, param)

    #     # breakpoint()


    def update(self, state, action, nextState, reward):
        state_tensor = torch.from_numpy(state.state).float()
        nextState_tensor = torch.from_numpy(nextState.state).float()

        # action_onehot = [0] * self.size_out
        # action_onehot[action] = 1
        # action_onehot = torch.tensor(action_onehot)


        if len(self.loss_history):
            self.confidence = min(1 / np.mean(self.loss_history[-5:]), 1)
        else:
            self.confidence = 0

        pred_q = self.policy_net(state_tensor)
        target_q = torch.clone(pred_q) * 0
        # target_q[0][action] = reward + self.discount * torch.max(self.policy_net(nextState_tensor))

        # TODO: Confidendce of the network * self.discount max...
        target_q[0][action] = reward + self.confidence * self.discount * torch.max(self.target_net(nextState_tensor))
        # self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state.hash, action) )

        print(state_tensor)
        print(self.policy_net(state_tensor))
        print(reward)
        print(action)
        

        loss = self.criterion(pred_q, target_q)
        self.loss_history.append(loss.detach().numpy())

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        print(self.policy_net(state_tensor))
        print('_______________________LOSS____________________________', loss)


        # if reward < 15:
        #     breakpoint()


        print('_________________________________________')
        # Update the target network, copying all weights and biases in DQN
        self.target_iter += 1
        if self.target_iter == 4:
            self.target_iter = 0
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.scheduler.step()

