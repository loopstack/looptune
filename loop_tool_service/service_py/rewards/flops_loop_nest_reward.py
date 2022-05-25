from compiler_gym.spaces import Reward
import pdb

class Reward(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="flops_loop_nest",
            observation_spaces=["flops_loop_nest"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        print("Reward Flops: reset")
        del benchmark  # unused
        self.prev_flops = observation_view["flops_loop_nest"]

    def update(self, action, observations, observation_view):
        print("Reward Flops: update")
        del action
        del observation_view
        new_flops = observations[0]
        reward = float(self.prev_flops - new_flops) / self.prev_flops
        self.prev_flops = new_flops
        return reward


class RewardTensor(Reward):
    """An example reward that uses changes in the "flops" observation value
    to compute incremental reward.
    """

    def __init__(self):
        super().__init__(
            id="flops_loop_nest_tensor",
            observation_spaces=["flops_loop_nest_tensor"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.prev_flops = 0

    def reset(self, benchmark: str, observation_view):
        print("Reward Flops: reset")
        del benchmark  # unused
        self.prev_flops = observation_view["flops_loop_nest_tensor"]

    def update(self, action, observations, observation_view):
        print("Reward Flops: update")
        del action
        del observation_view
        
        new_flops = observations[0]
        reward = float(self.prev_flops - new_flops) / self.prev_flops
        self.prev_flops = new_flops        
        return reward
