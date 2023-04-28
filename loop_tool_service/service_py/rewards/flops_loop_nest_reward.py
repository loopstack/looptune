from compiler_gym.spaces import Reward
import pdb

class RewardScalar(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="flops_loop_nest",
            observation_spaces=["flops_loop_nest"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        # print("Reward flops_loop_nest: reset")
        del benchmark  # unused
        self.prev_flops = observation_view["flops_loop_nest"]

    def update(self, action, observations, observation_view):
        # print("Reward flops_loop_nest: update")
        del action
        del observation_view
        new_flops = observations[0]
        reward = float(new_flops - self.prev_flops)
        self.prev_flops = new_flops
        print(f'Reward = {reward}')
        return reward


class RewardTensor(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="flops_loop_nest_tensor",
            observation_spaces=["flops_loop_nest_tensor"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        # print("Reward flops_loop_nest_tensor: reset")
        del benchmark  # unused
        self.prev_flops = observation_view["flops_loop_nest_tensor"]

    def update(self, action, observations, observation_view):
        # print("Reward flops_loop_nest_tensor: update")
        del action
        del observation_view
        
        new_flops = observations[0]
        reward = float(new_flops - self.prev_flops)
        self.prev_flops = new_flops        
        return reward


class AbsoluteRewardTensor(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            name="flops_loop_nest_tensor",
            observation_spaces=["flops_loop_nest_tensor"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        # self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        # print("Reward flops_loop_nest_tensor: reset")
        del benchmark  # unused
        # self.prev_flops = observation_view["flops_loop_nest_tensor"]

    def update(self, action, observations, observation_view):
        # print("Reward flops_loop_nest_tensor: update")
        del action
        del observation_view     
        return observations[0]


import os 
class NormRewardTensor(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self, obs='flops_loop_nest_tensor'):
        self.obs = obs
        super().__init__(
            name=obs,
            observation_spaces=[obs],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.max_flops = float(os.getenv('MAX_GFLOPS'))
        self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.prev_flops = observation_view[self.obs]

    def update(self, action, observations, observation_view):
        del action
        del observation_view
        
        new_flops = observations[0]
        reward = float(new_flops - self.prev_flops) / self.max_flops
        self.prev_flops = new_flops        
        return reward