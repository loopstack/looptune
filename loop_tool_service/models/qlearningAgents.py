from networkx.drawing.nx_pydot import to_pydot

from cgi import test
from re import S
from turtle import title
from loop_tool_service.models.util import Counter
from loop_tool_service.models.learningAgents import ReinforcementAgent
from matplotlib import pyplot as plt

import random, math

import os
import numpy as np
import pdb
import networkx as nx

import json
class State():
  def __init__(self, hash, string):
    self.hash = hash
    self.string = string


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
        self.Q_counts = {}
        self.hash_string = {}

        self.train_history = []
        self.test_history = []
        self.history_policy = []
        self.state = None
        self.epoch_count = 10
        self.converged = False

    def plot_history(self):
        train_policy = self.history_policy[:self.numTraining//self.numTest] 
        
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].title.set_text('Train rewards')
        axs[0, 0].plot(self.train_history, color="red")
        axs[0, 0].plot(np.zeros_like(self.train_history), color="blue")

        axs[0, 1].title.set_text('Test rewards')
        axs[0, 1].plot(self.test_history, color="green")
        axs[0, 1].plot(np.zeros_like(self.test_history), color="blue")
        
        axs[1, 0].title.set_text('Train rewards')
        axs[1, 0].plot(train_policy, color="green")
        axs[1, 0].plot(np.zeros_like(train_policy), color="blue")
                
        # ax4.title.set_text('Test rewards')
        # ax4.plot(test_policy, color="green")
        # ax4.plot(np.zeros_like(test_policy), color="blue")
        
        plt.tight_layout()
        # plt.show()
        reward_file = "rewards.png"
        # if os.path.isfile(reward_file):
        #   os.remove(reward_file)
        plt.savefig(reward_file)

    def print_state(self, state):
        print(f"====================================================================")
        print(state.string)
        print(f"====================================================================")
        if state.hash not in self.Q:
            print('No Q table available')
            return
        print('Actions:')
        for a, prob in self.Q[state.hash].items():
            print(f'{self.env.action_space.to_string(a)} = {prob}')
        
        print("-------------------------------------------")
        if state.hash in self.Q_counts:
            for a, c in self.Q_counts[state.hash].items():
                print(f'{self.env.action_space.to_string(a)} = {c}')
                
        print(f"====================================================================")


    def print_policy(self):
        for tree_hash, state_string in self.hash_string.items():
            best_action = max(self.Q[tree_hash], key=self.Q[tree_hash].get)
            print("*******************************************************")
            print(f"Next Action = {self.actionSpace.to_string(best_action)}\n")
            # print(to_pydot(state).to_string())
            print(state_string)
          

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # pdb.set_trace()

        if state.hash not in self.Q.keys() or action not in self.Q[state.hash]:
            return 0.0

        return self.Q[state.hash][action]
        

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


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        if state.hash not in self.Q:
            return 0

        chosen_action = max(self.Q[state.hash], key=self.Q[state.hash].get)
        return chosen_action
        
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.exploration, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        available_actions = json.loads(self.env.send_param("available_actions", ""))
        print(f"Available_actions = {available_actions}")
        available_actions = [self.env.action_space.from_string(a) for a in available_actions]


        if state.hash not in self.Q_counts.keys():
            self.Q_counts[state.hash] = Counter()
            self.Q[state.hash] = Counter()

            for a in available_actions:
                self.Q_counts[state.hash][a] = 0
                self.Q[state.hash][a] = 0.0

                
        if random.random() < self.exploration:
            print('Explore <<<<<<<<<<<<<<<<<<<<<')
            chosen_action = min(self.Q_counts[state.hash], key=self.Q_counts[state.hash].get)

            if chosen_action not in available_actions:
                pdb.set_trace()
        else:
            print('Policy <<<<<<<<<<<<<<<<<<<<<')
            chosen_action = self.computeActionFromQValues(state)

        self.Q_counts[state.hash][chosen_action] += 1
        return chosen_action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # pdb.set_trace()

        if state.hash not in self.hash_string:
          self.hash_string[state.hash] = state.string
        
        self.Q[state.hash][action] += self.learning_rate * ( reward + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state, action) )


    def evalPolicy(self, save_history=False):
        if self.Q == {}:
            return
        print("<<<<< Evaluate Policy >>>>>>>")    
        e_copy = self.exploration
        self.exploration = 0
        actions = []
        rewards = []

        self.env.reset(benchmark=self.bench) 
        observation = self.env.observation[self.observation]
        state = State(string=observation, hash=observation)

        for i in range(self.numTest):
            self.print_state(state)
            action = self.computeActionFromQValues(state)
            observation, reward, done, info = self.env.step(
                action=action,
                observation_spaces=[self.observation],
                reward_spaces=[self.reward],
            )
            state = State(string=observation[0], hash=observation[0])
            actions.append(action)
            rewards.append(reward[0])


            if save_history:
                flops = self.env.observation[self.reward]
                self.test_history.append(flops/1e9)
                self.plot_history()


        # # if any("dummy" == self.env.action_space.to_string(a) for a in actions):
        # if all([ r < 0.05 for r in rewards[:10]]):
        #     # print(f'Rewards = {reward_history}')
        #     self.converged = True

        self.exploration = e_copy
        self.history_policy.append(self.env.observation[self.reward]/1e9)
        # self.print_policy()

        print('#################################################')
        print("Current best policy:")
        print(state.string)
        print('#################################################')
        
  

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def test(self):
        print("\n============================ TESTING ==================================\n")
        self.evalPolicy(save_history=True)
    

    def train(self, iterations=None):

        for i in range(self.numTraining):
            print(f"**************************** {i} ******************************")
            if i % self.numTest == 0:
                self.evalPolicy()
                if self.converged:
                    print("\n\nConverged!!!\n")
                    break

                print('\n^^^^ New Epoch ^^^^^^\n')
                self.env.reset(benchmark=self.bench)
                observation = self.env.observation[self.observation]
                state = State(string=observation, hash=observation)
                state_start = state
                self.startEpisode()                

            action = self.getAction(state)
            # action = self.env.action_space.from_string(actions[i])
            print(f"Chosen Action = {self.env.action_space.to_string(action)}\n")

            state_prev = state

            try:
                observation, rewards, done, info = self.env.step(
                    action=action,
                    observation_spaces=[self.observation],
                    reward_spaces=[self.reward],
                )
            except ServiceError as e:
                print(f"AGENT: Timeout Error Step: {e}")
                continue
            except ValueError:
                pdb.set_trace()
                pass
            # available_actions = info[""]

            state = State(string=observation[0], hash=observation[0])
            print(f"{rewards}\n")
            print(f"{info}\n")

            # agent.update(state_prev, action, state, rewards[0])
            self.observeTransition(state_prev, action, state, rewards[0])

            flops = self.env.observation[self.reward]
            self.train_history.append(flops/1e9)
            self.plot_history()

            self.print_state(state_prev)
            # pdb.set_trace()


            
            print('Action = ', action)

            print(f'Flops = {flops/1e9}')
            # pdb.set_trace()
            # print(f"Current speed = {state.FLOPS()} GFLOPS")

            self.stopEpisode()

        print(f"====================================================================")

        self.plot_history()
        # self.print_policy()
        print(f"====================================================================")


        # for k, v in self.Q.items(): 
        #     print(f'{k}')
        #     print('Actions:')
        #     for a, prob in v.items():
        #         print(f'{self.env.action_space.to_string(a)} = {prob}')
        #     print(f"====================================================================")
        # pdb.set_trace()

        print("============================ END ==================================")

        # print(f"Start speed = {state_start.loop_tree.FLOPS() } GFLOPS")
        # print(f"Final speed = {state.FLOPS()} GFLOPS")
