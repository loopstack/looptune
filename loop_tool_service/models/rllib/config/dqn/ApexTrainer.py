import ray
import os

def get_config(sweep=False):
    hiddens_layers = [10, 15]
    hiddens_width = [500, 1000]
    return {
        "log_level": "ERROR",
        "env": "compiler_gym", 
        "observation_space": "loops_tensor",
        "framework": 'torch',
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": ray.tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]) if sweep else [1000] * 10,
            # "post_fcnet_hiddens":
            # "fcnet_activation": 
            # "post_fcnet_activation":
            # "no_final_linear":
            # "free_log_std":
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1, #torch.cuda.device_count(),
        'num_workers': int(ray.cluster_resources()['CPU'] / 2  - 1),
        "rollout_fragment_length": 10, 
        "train_batch_size": 790, # train_batch_size == num_workers * rollout_fragment_length
        "explore": True,
        "gamma": ray.tune.uniform(0.9, 0.99) if sweep else 0.95,
        "lr": ray.tune.uniform(1e-6, 1e-8) if sweep else 1e-6,
        # DQN specific parameters
        # "replay_buffer_config": {
        #     "type": "MultiAgentPrioritizedReplayBuffer",
        #     "capacity": 50000,
        # },
        # "num_steps_sampled_before_learning_starts": 10000,
        # "exploration_config": {
        #     "epsilon_timesteps": 200000,
        #     "final_epsilon": 0.01
        # }
        
    }
