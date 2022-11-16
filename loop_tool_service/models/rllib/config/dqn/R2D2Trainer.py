#     action, state, extra = policy.compute_single_action(
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/policy/policy.py", line 215, in compute_single_action
#     out = self.compute_actions_from_input_dict(
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/policy/torch_policy.py", line 294, in compute_actions_from_input_dict
#     return self._compute_action_helper(input_dict, state_batches,
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/utils/threading.py", line 21, in wrapper
#     return func(self, *a, **k)
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/policy/torch_policy.py", line 908, in _compute_action_helper
#     self.action_distribution_fn(
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/agents/dqn/r2d2_tf_policy.py", line 258, in get_distribution_inputs_and_class
#     q_vals, logits, probs_or_logits, state_out = func(
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/agents/dqn/dqn_torch_policy.py", line 350, in compute_q_values
#     model_out, state = model(input_dict, state_batches or [], seq_lens)
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/models/modelv2.py", line 244, in __call__
#     res = self.forward(restored, state or [], seq_lens)
#   File "/private/home/dejang/.conda/envs/compiler_gym/lib/python3.8/site-packages/ray/rllib/models/torch/recurrent_net.py", line 185, in forward
#     assert seq_lens is not None
# AssertionError

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
            # "use_attention": True,
            "use_lstm": True,
            "max_seq_len": 20,
            "attention_use_n_prev_actions": 20,
            "attention_use_n_prev_rewards": 20,
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
        'num_workers': int(ray.cluster_resources()['CPU']) // 2 - 1,
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
