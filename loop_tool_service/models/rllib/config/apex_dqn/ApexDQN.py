import ray
import os

def get_config(sweep=False):
    hiddens_layers = [10, 15]
    hiddens_width = [500, 1000]
    num_workers = int(ray.cluster_resources()['CPU'] * 0.9 - 1)
    rollout_fragment_length = 5
    return {
        "log_level": "CRITICAL",
        "env": "compiler_gym", 
        "observation_space": "loops_tensor",
        "framework": 'torch',
        "model": {
            "vf_share_layers": True,
            "fcnet_hiddens": ray.tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]) if sweep else [512] * 6,
            # "post_fcnet_hiddens":
            # "fcnet_activation": 
            # "post_fcnet_activation":
            # "no_final_linear":
            # "free_log_std":
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1, #torch.cuda.device_count(),
        'num_workers': num_workers,
        "rollout_fragment_length": rollout_fragment_length, 
        "train_batch_size": num_workers * rollout_fragment_length, # train_batch_size == num_workers * rollout_fragment_length
        "explore": True,
        "gamma": ray.tune.uniform(0.9, 0.99) if sweep else 0.95,
        "lr": ray.tune.uniform(1e-6, 1e-8) if sweep else 1e-6,        
    }
