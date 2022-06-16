import random,time
import pdb


class ValueEstimationAgent():

    def __init__(self, learning_rate=1.0, exploration=0.05, discount=0.8, numTraining = 20, numTest=10):
        """
        learning_rate    - learning rate
        exploration  - exploration rate
        discount    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.learning_rate = float(learning_rate)
        self.exploration = float(exploration)
        self.discount = float(discount)
        self.numTraining = int(numTraining)
        self.numTest = int(numTest)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        raise NotImplementedError

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        raise NotImplementedError

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        raise NotImplementedError

    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        raise NotImplementedError

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update(state, action, nextState, deltaReward)
                      which you should override.
        - Use self.getLegalActions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        raise NotImplementedError

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return [ a for a in range(self.actionSpace.n)]

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
            self.accumTrainRewards_history.append(self.accumTrainRewards)
        else:
            self.accumTestRewards += self.episodeRewards
            self.accumTestRewards_history.append(self.accumTestRewards)

        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.exploration = 0.0    # no exploration
            self.learning_rate = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, env, bench, observation, reward, numTraining=100, numTest=10, exploration=0.5, learning_rate=0.5, discount=1):
        """
        learning_rate    - learning rate
        exploration  - exploration rate
        discount    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.env = env
        self.bench = bench
        self.observation = observation
        self.reward = reward
        
        self.actionSpace = env.env.action_space
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.accumTrainRewards_history = []
        self.accumTestRewards_history = []
        
        self.numTraining = int(numTraining)
        self.numTest = int(numTest)
        
        self.exploration = float(exploration)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, exploration):
        self.exploration = exploration

    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def setDiscount(self, discount):
        self.discount = discount

    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print(('Beginning %d episodes of Training' % (self.numTraining)))

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print ('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print(('\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining)))
                print(('\tAverage Rewards over all training: %.2f' % (
                        trainAvg)))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print(('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)))
                print(('\tAverage Rewards over testing: %.2f' % testAvg))
            print(('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg)))
            print(('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off exploration and learning_rate)'
            print(('%s\n%s' % (msg,'-' * len(msg))))
