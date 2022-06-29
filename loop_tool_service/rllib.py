# Print the versions of the libraries that we are using:
import compiler_gym
import ray

from ray.rllib.agents.ppo import PPOTrainer
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit
from ray import tune
from itertools import islice
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.util.registration import register

import loop_tool_service

from service_py.datasets import loop_tool_dataset
from service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward

import pdb

# def register_env():
#     register(
#         id="loop_tool_env-v0",
#         entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
#         kwargs={
#             "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
#             "rewards": [
#                 # flops_loop_nest_reward.RewardTensor(),
#                 runtime_reward.RewardTensor(),
#                 ],
#             "datasets": [
#                 loop_tool_dataset.Dataset(),
#             ],
#         },
#     )

# register_env()


def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest",
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


# tune.register_env("compiler_gym", make_training_env)

# Lets cycle through a few calls to reset() to demonstrate that this environment
# selects a new benchmark for each episode.
# with make_training_env() as env:
#     pdb.set_trace()
#     env.reset()
#     print(env.benchmark)
#     env.reset()
#     print(env.benchmark)
#     env.reset()
#     print(env.benchmark)

# (Re)Start the ray runtime.
if ray.is_initialized():
    ray.shutdown()
ray.init(include_dashboard=False, ignore_reinit_error=True)

tune.register_env("compiler_gym", make_training_env)

print("hack111:")

analysis = tune.run(
    PPOTrainer,
    fail_fast="raise",
    checkpoint_at_end=True,
    stop={
        "episodes_total": 5,
    },
    config={
        "seed": 0xCC,
        "num_workers": 1,
        # Specify the environment to use, where "compiler_gym" is the name we
        # passed to tune.register_env().
        "env": "compiler_gym",
        # Reduce the size of the batch/trajectory lengths to match our short
        # training run.
        "framework":'tf2',
        "disable_env_checking":True,
        "rollout_fragment_length": 5,
        "train_batch_size": 5,
        "sgd_minibatch_size": 5,
    }
)
print("hack222:")

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
breakpoint()

# Lets define a helper function to make it easy to evaluate the agent's
# performance on a set of benchmarks.

def run_agent_on_benchmarks(benchmarks):
    """Run agent on a list of benchmarks and return a list of cumulative rewards."""
    with make_env() as env:
        rewards = []
        for i, benchmark in enumerate(benchmarks, start=1):
            observation, done = env.reset(benchmark=benchmark), False
            while not done:
                action = agent.compute_single_action(observation)
                observation, _, done, _ = env.step(int(action))
            rewards.append(env.episode_reward)
            
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

pdb.set_trace()
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