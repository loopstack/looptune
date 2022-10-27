import ray
import os

def get_config(sweep=False):
    hiddens_layers = [8, 16]
    hiddens_width = [500, 1000]
    return {
        "log_level": "ERROR",
        "env": "compiler_gym", 
        "observation_space": "loops_tensor",
        "framework": 'torch',
        "model": {
            "custom_model": "my_model",
            "custom_model_config": {"action_mask": [1, 1, 1, 1]},
            "vf_share_layers": True,
            "fcnet_hiddens": ray.tune.choice([ [w] * l for w in hiddens_width for l in hiddens_layers ]) if sweep else [1000] * 8,
            # "post_fcnet_hiddens":
            # "fcnet_activation": 
            # "post_fcnet_activation":
            # "no_final_linear":
            # "free_log_std":
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), #torch.cuda.device_count(),
        'num_workers': int(ray.cluster_resources()['CPU']) - 1 - 1,
        "rollout_fragment_length": 10, 
        "train_batch_size": 790, # train_batch_size == num_workers * rollout_fragment_length
        "num_sgd_iter": 30,
        # "evaluation_interval": 5, # num of training iter between evaluations
        # "evaluation_duration": 10, # num of episodes run per evaluation period
        "explore": True,
        "gamma": ray.tune.uniform(0.7, 0.99) if sweep else 0.95,
        "lr": ray.tune.uniform(1e-6, 1e-8) if sweep else 3.847293324197388e-3,
    }
