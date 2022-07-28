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

import ray
from ray import tune
# from ray.rllib.algorithms import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register
from compiler_gym.wrappers import TimeLimit

import loop_tool_service

from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward



import wandb
from ray.tune.integration.wandb import WandbLoggerCallback

# wandb.tensorboard.patch(root_logdir="...")
# wandb.init(project="loop_tool_agent", entity="dejang")



tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=5, help="Number of iterations to train."
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
                flops_loop_nest_reward.RewardTensor(),
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
    train_benchmarks = bench
    val_benchmarks = bench
    test_benchmarks = bench

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





class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    # ray.init(local_mode=args.local_mode)
    ray.init(ignore_reinit_error=True)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.framework == "torch" else CustomModel
    )

    config = {
        "env": "compiler_gym",  # or "corridor" if registered above
        "env_config": {
            "corridor_length": 5,
        },
        "num_workers": 75,  # parallelism
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "framework": args.framework,
        # define search space here
        # "parameter_1": tune.choice([1, 2, 3]),
        # "parameter_2": tune.choice([4, 5, 6]),
    }

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


    agent = PPOTrainer(
        env="compiler_gym",
        config={
            "num_workers": 1,
            "seed": 0xCC,
            # For inference we disable the stocastic exploration that is used during
            # training.
            "explore": False,
        },
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
    # agent.restore(checkpoint)
    print("hack555:")
    # breakpoint()

    # Lets define a helper function to make it easy to evaluate the agent's
    # performance on a set of benchmarks.

    def run_agent_on_benchmarks(benchmarks):
        """Run agent on a list of benchmarks and return a list of cumulative rewards."""
        with make_env() as env:
            rewards = []
            flops = 0

            for i, benchmark in enumerate(benchmarks, start=1):
                observation, done = env.reset(benchmark=benchmark), False
                step_count = 0

                while not done:
                    action = agent.compute_single_action(observation)
                    observation, _, done, _ = env.step(int(action))
                    flops = env.observation["flops_loop_nest_tensor"]
                    step_count += 1
                    print(f'{step_count}. Flops = {flops}, Actions = {env.actions}')

                walk_count = 10
                search_depth=0
                search_width = 10000
                # breakpoint()
                reward_actions_str = env.send_param("search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
                print(f'Search = {reward_actions_str}')
                reward_actions = json.loads(reward_actions_str)
                # breakpoint()
                rewards.append(flops / reward_actions[0])
                
                print(f"[{i}/{len(benchmarks)}] ")

        return rewards

    breakpoint()
    # Evaluate agent performance on the validation set.
    val_rewards = run_agent_on_benchmarks(val_benchmarks)

    # Evaluate agent performance on the holdout test set.
    test_rewards = run_agent_on_benchmarks(test_benchmarks)

    print("hack888")
    # Finally lets plot our results to see how we did!
    from matplotlib import pyplot as plt
    import numpy as np

    # def plot_results(x, y, name, ax):
    #     plt.sca(ax)
    #     plt.bar(range(len(y)), y)
    #     plt.ylabel("Reward (higher is better)")
    #     plt.xticks(range(len(x)), x, rotation=90)
    #     plt.title(f"Performance on {name} set")


    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(13, 3)
    # plot_results(val_benchmarks, val_rewards, "val", ax1)
    # plot_results(test_benchmarks, test_rewards, "test", ax2)
    # # plt.show()
    # plt.savefig('bench.png')

    # def plot_history(self):        

    # breakpoint()
    fig, axs = plt.subplots(1, 2)

    axs[0].title.set_text('Train rewards')
    axs[0].plot(val_rewards, color="red")
    axs[0].plot(np.zeros_like(val_rewards), color="blue")

    axs[1].title.set_text('Test rewards')
    axs[1].plot(test_rewards, color="green")
    axs[1].plot(np.zeros_like(test_rewards), color="blue")

    plt.tight_layout()
    # plt.show()
    reward_file = "bench.png"
    plt.savefig(reward_file)


    
    ray.shutdown()
