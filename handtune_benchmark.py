import loop_tool as lt
import sys
import re
import json

import loop_tool_service
from loop_tool_service.paths import BENCHMARKS_PATH, LOOP_TOOL_ROOT
from loop_tool_service.service_py.rewards import flops_loop_nest_reward, flops_reward, runtime_reward
from loop_tool_service.service_py.datasets import loop_tool_dataset, loop_tool_test_dataset

import compiler_gym
from compiler_gym.util.registration import register
from compiler_gym.wrappers import TimeLimit

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
import loop_tool_service.models.rllib.my_net_rl as my_net_rl 

import torch

import os 


def register_env():
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardTensor(),
                # flops_loop_nest_reward.AbsoluteRewardTensor(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
                loop_tool_test_dataset.Dataset()
            ],
        },
    )


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

def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    """Make a reinforcement learning environment that cycles over the
    set of training benchmarks in use.
    """
    del args  # Unused env_config argument passed by ray
    return make_env()


def get_search_lt(
    target_uri, 
    walk_count=10,
    step_count=10,
    search_depth=0,
    search_width=10
    ):
    with make_env() as env:
        for full_bench_uri in env.datasets.benchmark_uris(): 
            if target_uri in full_bench_uri:
                full_target_uri = full_bench_uri


        env.reset(benchmark=full_target_uri)
        reward_actions_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
        print(f'Search = {reward_actions_str}')
        reward_actions = json.loads(reward_actions_str)

        actions = [ env.action_space.from_string(a_str) for a_str in reward_actions[1]]
        env.multistep(actions=actions)
        env.send_param("print_looptree", "")
        final_flops = env.observation["flops_loop_nest_tensor"]
        print(f'Final = {final_flops} GFlops')

        if abs(float(reward_actions[0]) - final_flops) > 0.15 * final_flops: breakpoint()



def get_network_lt(target_uri, checkpoint_path, step_count=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    last_run_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"
    with open(last_run_path/"config.json", 'r') as f: config = json.load(f)

    ModelCatalog.register_custom_model(
        "my_model", my_net_rl.TorchCustomModel
    )
    
    config['explore'] = False
    agent = PPOTrainer(
        env="compiler_gym",
        config=config
    )
    agent.restore(checkpoint_path)
    policy = agent.get_policy()

    with make_env() as env:
        for full_bench_uri in env.datasets.benchmark_uris(): 
            if target_uri in full_bench_uri:
                full_target_uri = full_bench_uri

    observation, done = env.reset(benchmark=full_target_uri), False
    
    for i in range(step_count):
        logits, _ = policy.model({"obs": torch.Tensor(observation).to(device)})
        sorted_actions_q, sorted_actions = torch.sort(logits, descending=True)
        for action in sorted_actions.flatten().tolist():
            observation, _, done, info = env.step(int(action))
            if not info['action_had_no_effect']:
                break

    env.send_param("print_looptree", "")
    print(f'Final = {env.observation["flops_loop_nest_tensor"]} Flops')



def handtune_benchmark(target_uri):
    file = list(BENCHMARKS_PATH.glob(f'**/{target_uri}.txt'))[0]
    print(file)
    with open(file, 'r') as f:
        ir = lt.deserialize(f.read())

    tree = lt.LoopTree(ir)
    print(tree)

    with lt.Backend("loop_nest"): 
        print(f'Speed = {tree.FLOPS() / 1e9} GFLOPS')

    def mm(A, B):
        s = lt.SymbolGenerator()
        C = A.to(s.m, s.k) * B.to(s.k, s.n)
        return C.sum(s.k)
    
    m, n, k = [int(x) for x in re.findall('[0-9]+', target_uri)][:3]
    A = lt.Tensor(m, k)
    B = lt.Tensor(k, n)

    C = mm(A, B)

    C.set(tree)
    with lt.Backend("loop_nest"): 
        C = lt.ui(C, "/tmp/woo.c")


        

if __name__ == '__main__':

    if len(sys.argv) > 2:
        print('Format: python handtune_benchmark.py benchmark_uri')
        exit()

    target_uri = sys.argv[1]
    checkpoint_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts/best_checkpoint"

    register_env()
    tune.register_env("compiler_gym", make_training_env)

    get_search_lt(target_uri)
    breakpoint()
    get_network_lt(target_uri, checkpoint_path)
    breakpoint()
    handtune_benchmark(target_uri)

