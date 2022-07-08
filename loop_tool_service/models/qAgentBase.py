import random,time
import pdb
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path


import loop_tool as lt

class State:
    def __init__(self, state, state_hash):
        self.state = state
        self.hash = state_hash

class QAgentBase():
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getAvailableActions(state) to know which actions
                      are available in a state
    """
    def __init__(self, 
            env, 
            bench,
            observation,
            reward, 
            numTraining=100, 
            numTest=10, 
            exploration=0.5, 
            learning_rate=0.5, 
            discount=1,
            save_file_path = "rewards.png"
        ):

        self.env = env
        self.bench = bench
        self.observation = observation
        self.reward = reward
        
        self.reward_dict = {}
        self.Q_counts = {}
        self.exploration = float(exploration)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        
        self.numTraining = int(numTraining)
        self.numTest = int(numTest)
        self.train_history = []
        self.test_history = []
        self.epochs_history = []
        self.loss_history = []
        self.converged = False
        self.save_file_path = Path(os.getenv('LOOP_TOOL_ROOT')) / Path('results') / Path(save_file_path)

    ####################################
    #    Override These Functions      #
    ####################################

    def hashState(self, state):
        raise NotImplementedError

    def getQValues(self, state) -> dict:
        raise NotImplementedError

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        raise NotImplementedError

    ####################################
    #    Read These Functions          #
    ####################################
    def getAvailableActions(self, state_hash) -> list:
        if state_hash in self.Q_counts:
            return list(self.Q_counts[state_hash].keys())

        available_actions = json.loads(self.env.send_param("available_actions", ""))
        print(f"Available_actions = {available_actions}")
        available_actions = [self.env.action_space.from_string(a) for a in available_actions]

        self.Q_counts[state_hash] = {}
        for a in available_actions:
            self.Q_counts[state_hash][a] = 0

        return available_actions
        

    def getBestQValue(self, state) -> float:  
        q_dict = self.getQValues(state) 
        return max(q_dict.values())

    def getBestQAction(self, state) -> float:  
        q_dict = self.getQValues(state) 
        return max(q_dict, key=q_dict.get)


    def getQValue(self, state, action) -> float:
        q_dict = self.getQValues(state) 
        if action not in q_dict:
            pdb.set_trace()
        return q_dict[action]


    def getAction(self, state, exploration):
        """
          Compute the action to take in the current state.  With
          probability self.exploration, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        available_actions = self.getAvailableActions(state.hash)

        if random.random() < exploration:
            print('Explore <<<<<<<<<<<<<<<<<<<<<')
            chosen_action = min(self.Q_counts[state.hash], key=self.Q_counts[state.hash].get)
            self.Q_counts[state.hash][chosen_action] += 1
        else:
            print('Policy <<<<<<<<<<<<<<<<<<<<<')
            chosen_action = self.getBestQAction(state)
        
        print(f"Chosen Action = {self.env.action_space.to_string(chosen_action)}\n")

        return chosen_action


    def evalPolicy(self, save_history=False):
        print("<<<<< Evaluate Policy >>>>>>>")    
        actions = []
        rewards = []

        self.env.reset(benchmark=self.bench) 
        obs = self.env.observation[self.observation]
        state = State(obs, self.hashState(obs))

        for i in range(self.numTest):
            self.print_state(state)
            action = self.getAction(state=state, exploration=0)

            if self.getQValue(state, action) <= 0:
                print("Stop! This action doesn't help")
                break # There is no known action that improves state

            # breakpoint()
            observation, reward, done, info = self.env.step(
                action=action,
                observation_spaces=[self.observation],
                reward_spaces=[self.reward],
            )
            state = State(observation[0], self.hashState(observation[0]))
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

        self.epochs_history.append(self.env.observation[self.reward]/1e9)
        # self.print_policy()

        print('#################################################')
        print("Current best policy:")
        self.env.send_param("print_looptree", "")
        print(f'Cumulative Reward = {sum(rewards)}')
        print('#################################################')        
        # breakpoint()


  
    def test(self):
        print("\n============================ TESTING ==================================\n")
        self.evalPolicy(save_history=True)


    def train(self, iterations=None):
        actions = [1, 3, 0, 0]
        my_actions = -1

        for i in range(self.numTraining):
            print(f"**************************** {i} ******************************")
            if i % self.numTest == 0:
                self.evalPolicy()
                if self.converged:
                    print("\n\nConverged!!!\n")
                    break

                print('\n^^^^ New Epoch ^^^^^^\n')
                self.env.reset(benchmark=self.bench)
                obs = self.env.observation[self.observation]
                state = State(obs, self.hashState(obs))


            # if i % 4 == 0:
            #     breakpoint()

            my_actions = (my_actions + 1) % 8
            if my_actions < 4 or True:
                action = actions[i % 4]
                available_actions = self.getAvailableActions(state.hash)
                self.Q_counts[state.hash][action] += 1
            else:
                action = self.getAction(state=state, exploration=self.exploration)

            # breakpoint()
            state_prev = state
            self.print_state(state)

            try:
                if state.hash not in self.reward_dict:
                    observation, rewards, done, info = self.env.step(action=action,observation_spaces=[self.observation],reward_spaces=[self.reward],)
                    self.reward_dict[state.hash] = rewards
                else:
                    observation, rewards, done, info = self.env.step(action=action,observation_spaces=[self.observation])
                    rewards = self.reward_dict[state.hash]
                    
            except ServiceError as e:
                print(f"AGENT: Timeout Error Step: {e}")
                pdb.set_trace()
            except ValueError:
                pdb.set_trace()
                pass
            
            # available_actions = info[""]
            state = State(observation[0], self.hashState(observation[0]))
            print(f"Reward = {rewards[0]}\n")
            # print(f"{info}\n")

            self.update(state_prev, action, state, rewards[0])

            flops = self.env.observation[self.reward]
            print(f'Flops = {flops/1e9}')

            self.train_history.append(flops/1e9)
            self.plot_history()


        print(f"====================================================================")

        self.plot_history()
        # self.print_policy()

        print("============================ END ==================================")


    def plot_history(self):        
        fig, axs = plt.subplots(2, 2)
        
        fig.suptitle(f'Obs = {self.observation}, Exp = {self.exploration}, Lr = {self.learning_rate}, Dis = {self.discount}')

        axs[0, 0].title.set_text('Train rewards')
        axs[0, 0].plot(self.train_history, color="red")
        axs[0, 0].plot(np.zeros_like(self.train_history), color="blue")

        axs[0, 1].title.set_text('Test rewards')
        axs[0, 1].plot(self.test_history, color="green")
        axs[0, 1].plot(np.zeros_like(self.test_history), color="blue")
        
        axs[1, 0].title.set_text('Epochs rewards')
        axs[1, 0].plot(self.epochs_history, color="green")
        axs[1, 0].plot(np.zeros_like(self.epochs_history), color="blue")
                
        axs[1, 1].title.set_text('Loss')
        axs[1, 1].plot(self.loss_history, color="green")
        axs[1, 1].plot(np.zeros_like(self.loss_history), color="blue")

        plt.tight_layout()
        self.save_file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.save_file_path)

    def print_state(self, state):
        print(f"====================================================================")
        self.env.send_param("print_looptree", "")
        print(f"====================================================================")
        print('Action Preference:')
        for a, prob in self.getQValues(state).items():
            print(f'{self.env.action_space.to_string(a)} = {prob}')

        print("-------------------------------------------")
        if state.hash in self.Q_counts:
            for a, c in self.Q_counts[state.hash].items():
                print(f'{self.env.action_space.to_string(a)} = {c}')
                
        print(f"====================================================================")


    # def print_policy(self):
    #     for tree_hash, state_string in self.hash_string.items():
    #         best_action = max(self.Q[tree_hash], key=self.Q[tree_hash].get)
    #         print("*******************************************************")
    #         print(f"Next Action = {self.env.action_space.to_string(best_action)}\n")
    #         # print(to_pydot(state).to_string())
    #         print(state_string)