# Print the versions of the libraries that we are using:
import argparse
import compiler_gym
import ray
import json

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit
from ray import tune
from itertools import islice
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register

import loop_tool_service

from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward

from my_net_rl import CustomModel, TorchCustomModel

import wandb
# wandb.tensorboard.patch(root_logdir="...")
wandb.init(project="loop_tool_agent", entity="dejang")




parser = argparse.ArgumentParser()

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)





args = parser.parse_args()
print(args.framework )


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
    reward_actions_str = env.send_param("greedy_search", f'{10}, {5}, {0}, {1000}')
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
ray.init(ignore_reinit_error=True)
tune.register_env("compiler_gym", make_training_env)

print("hack111:")
breakpoint()



ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel if args.framework == "torch" else CustomModel
    )



analysis = tune.run(
    PPOTrainer,
    fail_fast=True,
    reuse_actors=True,
    checkpoint_at_end=True,
    stop={
        "training_iteration": 10,
    },
    # resources_per_trial={"cpu": 64, "gpu": 8},

    config={
        "seed": 0xCC,
        "num_workers": 10,
        "num_gpus": 1, # tf2: <= 1
        # Specify the environment to use, where "compiler_gym" is the name we
        # passed to tune.register_env().
        "env": "compiler_gym",
        # Reduce the size of the batch/trajectory lengths to match our short
        # training run.
        "framework":'torch',
        # "model": {'fcnet_hiddens': [5, 5]},
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        # "framework":'tf2',
        # "eager_tracing": True,

        # 'log_level': 'DEBUG',
        "train_batch_size": 1000, # train_batch_size -> sgd_minibatch_size -> max_seq_len (x num_sgd_iter)
        "rollout_fragment_length": 10, # rollout_fragment_length < train_batch_size
        "sgd_minibatch_size": 128, # sgd_minibatch_size < train_batch_size
        "num_sgd_iter": 12,
        "shuffle_sequences": True,
        # "model": {'fcnet_hiddens': [200] * 5},
        "gamma": 0.8, #tune.grid_search([0.5, 0.8, 0.9]), # def 0.99
        "lr": 1e-4, #tune.grid_search([0.01, 0.001, 0.0001]), # def 1e-4
        # "horizon": 5, # (None) maximum timesteps an episode can last

        # "max_seq_len": 10 # Goes with LSTM max num steps
    }
)
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
        for i, benchmark in enumerate(benchmarks, start=1):
            observation, done = env.reset(benchmark=benchmark), False
            step_count = 0

            while not done:
                action = agent.compute_single_action(observation)
                observation, _, done, _ = env.step(int(action))
                step_count += 1

            walk_count = 10
            search_depth=0
            search_width = 10000
            # breakpoint()
            reward_actions_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
            print(reward_actions_str)
            reward_actions = json.loads(reward_actions_str)
            # breakpoint()
            rewards.append(env.episode_reward / reward_actions[0])
            
            print(f"[{i}/{len(benchmarks)}] ")

    return rewards


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


# If running in a notebook, finish the wandb run to upload the tensorboard logs to W&B
wandb.finish()
ray.shutdown()