# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from re import S
from loop_tool_service.models.util import Counter
from loop_tool_service.models.learningAgents import ReinforcementAgent

import random, math

import numpy as np
import pdb
import networkx as nx

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.exploration (exploration prob)
        - self.learning_rate (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.Q = {}


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        state_hash = nx.weisfeiler_lehman_graph_hash(state, node_attr='feature')

        if state_hash not in self.Q.keys() or action not in self.Q[state_hash]:
            return 0.0

        return self.Q[state_hash][action]
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"        

        if len(self.getLegalActions(state)) > 0:
            return max( [self.getQValue(state, action) for action in self.getLegalActions(state)] )
        return 0.0


    def computeActionFromQValues(self, state, legal_actions):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
            
        if len(legal_actions) > 0:                       
            q_dict = { action: self.getQValue(state, action) for action in legal_actions }
            return max(q_dict, key=q_dict.get)
        else:
            return None

    def getAction(self, state, available_actions):
        """
          Compute the action to take in the current state.  With
          probability self.exploration, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        if random.random() < self.exploration:
            action_str = random.choice(available_actions)
            return self.actionSpace.from_string(action_str)      
        else:
            action_str = self.computeActionFromQValues(state, legal_actions=available_actions)
            return self.actionSpace.from_string(action_str)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        state_hash = nx.weisfeiler_lehman_graph_hash(state, node_attr='feature')

        if state not in self.Q.keys():
            self.Q[state_hash] = Counter()
        
        self.Q[state_hash][action] += self.learning_rate * ( reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action) )
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, exploration=0.05,discount=0.8,learning_rate=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a exploration=0.1

        learning_rate    - learning rate
        exploration  - exploration rate
        discount    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['exploration'] = exploration
        args['discount'] = discount
        args['learning_rate'] = learning_rate
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


# class ApproximateQAgent(PacmanQAgent):
#     """
#        ApproximateQLearningAgent

#        You should only have to overwrite getQValue
#        and update.  All other QLearningAgent functions
#        should work as is.
#     """
#     def __init__(self, extractor='IdentityExtractor', **args):
#         self.featExtractor = util.lookup(extractor, globals())()
#         PacmanQAgent.__init__(self, **args)
#         self.weights = Counter()

#     def getWeights(self):
#         return self.weights

#     def getQValue(self, state, action):
#         """
#           Should return Q(state,action) = w * featureVector
#           where * is the dotProduct operator
#         """
#         "*** YOUR CODE HERE ***"
#         # pdb.set_trace()     
#         features = self.featExtractor.getFeatures(state, action)
#         result = 0
#         for feature in features.keys():
#             result += self.weights[feature] * features[feature]
#         return result

#     def update(self, state, action, nextState, reward):
#         """
#            Should update your weights based on transition
#         """
#         "*** YOUR CODE HERE ***"        
#         features = self.featExtractor.getFeatures(state, action)
#         diff = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
#         # pdb.set_trace()
#         for feature in features.keys():
#             self.weights[feature] += self.learning_rate * diff * features[feature]


#     def final(self, state):
#         "Called at the end of each game."
#         # call the super-class final method
#         PacmanQAgent.final(self, state)

#         # did we finish training?
#         if self.episodesSoFar == self.numTraining:
#             # you might want to print your weights here for debugging
#             "*** YOUR CODE HERE ***"
#             pass
