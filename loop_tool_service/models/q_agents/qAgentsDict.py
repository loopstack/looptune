from loop_tool_service.models.q_agents.qAgentBase import QAgentBase
import sys
import networkx as nx
import pickle
import numpy as np

class QAgentLoopTree(QAgentBase):
    def __init__(self, **args):
        QAgentBase.__init__(self, **args)
        # self.observation = "loop_tree"
        self.Q = {}


    def initQ(self, state):
        if state.hash not in self.Q:
            self.Q[state.hash] = {}
            available_actions = self.getAvailableActions(state.hash)
            for a in available_actions:
                self.Q[state.hash][a] = 0.0

    def hashState(self, state):
        if type(state) == str:
            return state
        elif type(state) in [np.array, np.ndarray]:
            return np.array2string(state)
        else:
            assert(0), 'hashState cannot make string out of state'

    def getQValues(self, state) -> dict:
        self.initQ(state)
        return self.Q[state.hash]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state)
        self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state, action) )



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
        if type(state) == type(None): breakpoint()
        state_unpickled = pickle.loads(state)
        nx.drawing.nx_pydot.write_dot(state_unpickled,sys.stdout)
        return nx.weisfeiler_lehman_graph_hash(state_unpickled, node_attr='feature_vector')

    def getQValues(self, state) -> dict:
        if type(state) == str:
            breakpoint()
        self.initQ(state.hash)
        return self.Q[state.hash]
        

    def update(self, state, action, nextState, reward):
        self.initQ(state)
        self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state, action) )

        # self.initQ(state.hash)
        # self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.getBestQValue(nextState) - self.getQValue(state.hash, action) )
