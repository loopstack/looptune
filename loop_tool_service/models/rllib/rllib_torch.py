"""
Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random
import json
from matplotlib import pyplot as plt


import ray
from ray import tune
# from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register
from compiler_gym.wrappers import TimeLimit

import loop_tool_service

from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
import my_net_rl 


import wandb
from ray.tune.integration.wandb import WandbLoggerCallback
from loop_tool_service.paths import LOOP_TOOL_ROOT

# wandb.tensorboard.patch(root_logdir="...")
# wandb.init(project="loop_tool_agent", entity="dejang")



# tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    default="torch",
    choices=["tf", "tf2", "tfe", "torch"],
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=20, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=100, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    default=False,
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    default=False,
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


def register_env():
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                # flops_loop_nest_reward.RewardTensor(),
                flops_loop_nest_reward.AbsoluteRewardTensor(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
            ],
        },
    )

register_env()

def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
        # reward_space="runtime",
    )

    env = TimeLimit(env, max_episode_steps=10)
    return env


# Let's create an environment and print a few attributes just to check that we
# have everything set up the way that we would like.
with make_env() as env:
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("Reward space:", env.reward_space)
    env.reset()
    reward_actions_str = env.send_param("search", f'{10}, {5}, {0}, {1000}')
    print(f"Best reward = {reward_actions_str}")

with make_env() as env:
    # The two datasets we will be using:
    lt_dataset = env.datasets["loop_tool_simple-v0"]
    # train_benchmarks = list(islice(lt_dataset.benchmarks(), 1))
    # train_benchmarks, val_benchmarks = train_benchmarks[:2], train_benchmarks[2:]
    # test_benchmarks = list(islice(lt_dataset.benchmarks(), 2))
    
    bench = ["benchmark://loop_tool_simple-v0/simple",
             "benchmark://loop_tool_simple-v0/mm128", 
             "benchmark://loop_tool_simple-v0/mm"] 
    train_benchmarks = bench[0:]
    val_benchmarks = bench[0:]
    test_benchmarks = bench[0:]

print("Number of benchmarks for training:", len(train_benchmarks))
print("Number of benchmarks for validation:", len(val_benchmarks))
print("Number of benchmarks for testing:", len(test_benchmarks))

def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    """Make a reinforcement learning environment that cycles over the
    set of training benchmarks in use.
    """
    del args  # Unused env_config argument passed by ray
    return CycleOverBenchmarks(make_env(), train_benchmarks)


# (Re)Start the ray runtime.
if ray.is_initialized():
    ray.shutdown()


tune.register_env("compiler_gym", make_training_env)




if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    print(f"Running with following CLI options: {args}")

    # ray.init(local_mode=args.local_mode)
    ray.init(local_mode=args.local_mode, ignore_reinit_error=True)

    # # Can also register the env creator function explicitly with:
    ModelCatalog.register_custom_model(
        "my_model", my_net_rl.TorchCustomModel# if args.framework == "torch" else my_net_rl.CustomModel
    )

    config = {
        "env": "compiler_gym",  # or "corridor" if registered above
        "env_config": {
            "corridor_length": 5,
        },
        "num_workers": 75,  # parallelism
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        # "model": {"fcnet_hiddens": [100] * 4},
        
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "framework": args.framework,
        "train_batch_size": 1000,
        "evaluation_interval": 5, # num of training iter between evaluations
        "evaluation_duration": 10, # num of episodes run per evaluation period
        # define search space here
        # "parameter_1": tune.choice([1, 2, 3]),
        # "parameter_2": tune.choice([4, 5, 6]),
    }
    device = 'cuda' if config['num_workers'] > 0 else 'cpu'

    stop = {
        "training_iteration": args.stop_iters,
        # "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    # if args.no_tune:
        # # manual training with train loop using PPO and fixed learning rate
        # if args.run != "PPO":
        #     raise ValueError("Only support --run PPO with --no-tune.")
        # print("Running manual train loop without Ray Tune.")
        # ppo_config = ppo.DEFAULT_CONFIG.copy()
        # ppo_config.update(config)
        # # use fixed learning rate instead of grid search (needs tune)
        # ppo_config["lr"] = 1e-3
        # trainer = ppo.PPO(config=ppo_config, env="compiler_gym")
        # # run manual training loop and print results after each iteration
        # for _ in range(args.stop_iters):
        #     result = trainer.train()
        #     print(pretty_print(result))
        #     # stop training of the target train steps or reward are reached
        #     if (
        #         result["timesteps_total"] >= args.stop_timesteps
        #         or result["episode_reward_mean"] >= args.stop_reward
        #     ):
        #         break
    # else:
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    from ray.rllib.agents.ppo import PPOTrainer

    # breakpoint()    
    analysis = tune.run(
        # args.run, 
        PPOTrainer,
        reuse_actors=True,
        checkpoint_at_end=True,
        config=config, 
        stop=stop,         
        callbacks=[ WandbLoggerCallback(
                        project="loop_tool_agent",
                        # save_checkpoints=True,
                        api_key_file="/private/home/dejang/tools/loop_tool_env/loop_tool_service/models/rllib/wandb_key.txt",
                        log_config=False) ])

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(analysis, args.stop_reward)



    print("hack222:")
    # breakpoint()

    config['explore'] = False
    agent = PPOTrainer(
        env="compiler_gym",
        config=config
    )
    print("hack333:")
    # We only made a single checkpoint at the end of training, so restore that. In
    # practice we may have many checkpoints that we will select from using
    # performance on the validation set.
    checkpoint = analysis.get_best_checkpoint(
        metric="episode_reward_mean",
        mode="max",
        trial=analysis.trials[0]
    )
    print("hack444:")
    agent.restore(checkpoint)
    print("hack555:")
    
    
    import pandas as pd
    # Lets define a helper function to make it easy to evaluate the agent's
    # performance on a set of benchmarks.
    def run_agent_on_benchmarks(benchmarks):
        """Run agent on a list of benchmarks and return a list of cumulative rewards."""
        with make_env() as env:
            df_gflops = pd.DataFrame(np.zeros((len(benchmarks), 4)), columns=['bench', 'base', 'network', 'search'])

            flops = 0

            for i, benchmark in enumerate(benchmarks, start=0):
                observation, done = env.reset(benchmark=benchmark), False
                step_count = 0
                policy = agent.get_policy()
                print(policy.model.framework)

                df_gflops.loc[i, 'bench'] = benchmark
                df_gflops.loc[i, 'base'] = env.observation["flops_loop_nest_tensor"]/1e9

                # breakpoint()
                
                while not done:
                    env.send_param("print_looptree", "")
                    logits, _ = policy.model({"obs": torch.Tensor(observation).to(device)})
                    sorted_actions_q, sorted_actions = torch.sort(logits, descending=True)

                    assert (agent.compute_single_action(observation) == sorted_actions[0][0].item())
                    for q, a in zip(sorted_actions_q.flatten().tolist(), sorted_actions.flatten().tolist()):
                        print(env.action_space.to_string(a), q)
                        
                    for action in sorted_actions.flatten().tolist():
                        observation, _, done, info = env.step(int(action))
                        if not info['action_had_no_effect']:
                            break
                
                    flops = env.observation["flops_loop_nest_tensor"]
                    df_gflops.loc[i, 'network'] = flops/1e9
                    step_count += 1
                    print(f'{step_count}. Flops = {flops}, Actions = {[ env.action_space.to_string(a) for a in env.actions]}')

                walk_count = 10
                search_depth=0
                search_width = 10000
                # breakpoint()
                reward_actions_str = env.send_param("search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
                print(f'Search = {reward_actions_str}')
                reward_actions = json.loads(reward_actions_str)
                # breakpoint()
                df_gflops.loc[i, 'search'] = reward_actions[0]

                print(f"[{i}/{len(benchmarks)}] ")
        
        return df_gflops


    # Evaluate agent performance on the validation set.
    df_gflops_val = run_agent_on_benchmarks(val_benchmarks)

    # Evaluate agent performance on the holdout test set.
    df_gflops_test = run_agent_on_benchmarks(test_benchmarks)

    print("hack888")
    # Finally lets plot our results to see how we did!

    fig, axs = plt.subplots(1, 2)
    axs[0] = df_gflops_val.plot(x='bench', y=['base', 'network', 'search'], kind='bar', figsize=(9, 8), ax=axs[0])
    axs[1] = df_gflops_test.plot(x='bench', y=['base', 'network', 'search'], kind='bar', figsize=(9, 8), ax=axs[1])
    fig.savefig(LOOP_TOOL_ROOT/"bench.png")


    ray.shutdown()
