import argparse
import json
import torch
from tqdm import tqdm

import compiler_gym
from compiler_gym.wrappers import CycleOverBenchmarks
from compiler_gym.wrappers import TimeLimit

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

import loop_tool_service
# from loop_tool_service.models.rllib.rllib_torch import load_datasets, make_env
import loop_tool_service.models.rllib.my_net_rl as my_net_rl
from loop_tool_service.paths import LOOP_TOOL_ROOT
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from loop_tool_service.paths import BENCHMARKS_PATH, LOOP_TOOL_ROOT
import loop_tool as lt

import re


weights_path = LOOP_TOOL_ROOT/"loop_tool_service/models/weights"

# Training settings
parser = argparse.ArgumentParser(description="LoopTool Optimizer")
parser.add_argument("--policy", type=str, nargs='?', const=f"{weights_path}/policy.pt", default='', help="Path to the RLlib optimized network.")
parser.add_argument("--cost", type=str, nargs='?', const=f"{weights_path}/cost.pt", default='', help="Path to the cost model network.")
parser.add_argument("--benchmark", type=str, nargs='?', const='benchmark://loop_tool_test-v0/mm_127x127x127_36_40_20', default='benchmark://loop_tool_test-v0', help="Benchmark to run the search")
parser.add_argument(
    "--debug",
    default=False,
    action="store_true",
    help="Debuging",
)
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )
    env = TimeLimit(env, max_episode_steps=10)
    return env
    
def load_datasets(env, benchmark):
    lt_dataset = env.datasets[benchmark]
    data_size = 10 if args.debug else len(lt_dataset)
    benchmarks = list(lt_dataset.benchmarks())[:data_size]
    
    train_perc = 0.8
    train_size = int(train_perc * len(benchmarks))
    test_size = len(benchmarks) - train_size
    train_benchmarks, val_benchmarks = torch.utils.data.random_split(benchmarks, [train_size, test_size])

    print("Number of benchmarks for training:", len(train_benchmarks))
    print("Number of benchmarks for validation:", len(val_benchmarks))    
    return train_benchmarks, val_benchmarks



def load_policy_network(checkpoint_path):
    ModelCatalog.register_custom_model(
        "my_model", my_net_rl.TorchCustomModel
    )
    with open(checkpoint_path + '/config.json', 'r') as f:
        config = json.load(f)

    config["explore"] = False
    agent = PPOTrainer(
        env="compiler_gym",
        config=config
    )

    agent.restore(checkpoint_path)
    return agent.get_policy()


def load_value_network(network_path):
    return torch.jit.load(network_path).to(device)
    

def predict_optimal_actions(env, benchmark, policy, value_network):
    terminate_states = []
    actions = []
    observation, done = env.reset(benchmark=benchmark), False

    logits, _ = policy.model({"obs": torch.Tensor(observation).to(device)})
    sorted_actions_q, sorted_actions = torch.sort(logits, descending=True)

    if all(sorted_actions_q < 0):
        value_network(observation)
        terminate_states.append([actions])

    for q, a in zip(sorted_actions_q.flatten().tolist(), sorted_actions.flatten().tolist()):
        print(env.action_space.to_string(a), q)
        
    for action in sorted_actions.flatten().tolist():
        observation, _, done, info = env.step(int(action))
        if not info['action_had_no_effect']:
            break


def base_performance(env, benchmark):
    env.reset(benchmark=benchmark)
    return env.observation["flops_loop_nest_tensor"][0]

def greedy_search(
    env,
    benchmark, 
    walk_count=10,
    step_count=10,
    search_depth=0,
    search_width=10
):
    env.reset(benchmark=benchmark)
    actions_reward_str = env.send_param("greedy_search", f'{walk_count}, {step_count}, {search_depth}, {search_width}')
    actions_reward = json.loads(actions_reward_str)
    print(f'Greedy Search = {actions_reward}')
    return actions_reward[1]

def cost_search(env, benchmark):
    if args.cost == '': return 0
    env.send_param('load_cost_model', args.cost)
    reward_greedy = greedy_search(env, benchmark)
    env.send_param('load_cost_model', '')
    return reward_greedy
     
def policy_search(env, benchmark, search_depth=100, solutions=3):
    if args.policy == '': return 0
    env.reset(benchmark=benchmark)    
    env.send_param('load_policy_model', args.policy)
    actions_reward = json.loads(env.send_param("policy_search", f'{search_depth}, {solutions}'))
    env.send_param('load_policy_model', '')
    print(f'Policy Search = {actions_reward}')
    return actions_reward[1]


def cost_policy_search(env, benchmark, search_depth=100, solutions=3):
    if args.cost == '' or args.policy == '': return 0
    env.reset(benchmark=benchmark)
    env.send_param('load_cost_model', args.cost)
    env.send_param('load_policy_model', args.policy)
    actions_reward = json.loads(env.send_param("policy_search", f'{search_depth}, {solutions}'))
    env.send_param('load_cost_model', '')
    env.send_param('load_policy_model', '')
    print(f'Cost Policy Search = {actions_reward}')
    return actions_reward[1]


def handtune_benchmark(benchmark):
    bench_name = benchmark.split('/')[-1]
    file = list(BENCHMARKS_PATH.glob(f'**/{bench_name}.txt'))[0]
    print(file)
    with open(file, 'r') as f:
        ir = lt.deserialize(f.read())

    tree = lt.LoopTree(ir)
    print(tree)


    def mm(A, B):
        s = lt.SymbolGenerator()
        C = A.to(s.m, s.k) * B.to(s.k, s.n)
        return C.sum(s.k)
    
    m, n, k = [int(x) for x in re.findall('[0-9]+', bench_name)][:3]
    A = lt.Tensor(m, k)
    B = lt.Tensor(k, n)

    C = mm(A, B)

    C.set(tree)
    with lt.Backend("loop_nest"): 
        C = lt.ui(C, "/tmp/woo.c")

    return C.loop_tree.FLOPS() / 1e9



def plot_results(df_gflops_list):
    if len(df_gflops_list) == 1:
        df_gflops = df_gflops_list[0]
        fig, ax = plt.subplots(1, 1)
        ax = df_gflops.plot(x=df_gflops.columns[0], y=df_gflops.columns[1:], kind='bar', ax=ax)
    else:    
        width_ratio = int(len(df_gflops_list[0]) / len(df_gflops_list[1]))
        fig, axs = plt.subplots(1, len(df_gflops_list), figsize=(40, 5), gridspec_kw={'width_ratios': [width_ratio, 1]})
        for i, df_gflops in enumerate(df_gflops_list):
            axs[i] = df_gflops.plot(x=df_gflops.columns[0], y=df_gflops.columns[1:], kind='bar', ax=axs[i])
    
    fig.suptitle(f'GFlops comparison benchmarks', fontsize=16)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(f'{LOOP_TOOL_ROOT}/loop_tool_service/demos/demo.png')
       

def eval_benchmark(env, benchmark):
    print(benchmark)

    reward_base = base_performance(env, benchmark)
    reward_greedy = greedy_search(env, benchmark)
    reward_cost = cost_search(env, benchmark)
    reward_policy = policy_search(env, benchmark)
    reward_cost_policy = cost_policy_search(env, benchmark)
    reward_handtune = handtune_benchmark(benchmark)
    df_gflops = pd.DataFrame([[benchmark, reward_base, reward_greedy, reward_cost, reward_policy, reward_cost_policy, reward_handtune]], \
                      columns=['bench', 'base', 'greedy', 'cost', 'policy', 'cost_policy', 'handtune'])
    plot_results([df_gflops])


def eval_benchmarks(env, dataset):
    df_gflops_list = []
    df_gflops = pd.DataFrame(columns=['bench', 'base', 'greedy', 'policy'])
    train_benchmarks, val_benchmarks = load_datasets(env, dataset)
    for benchmarks in [train_benchmarks, val_benchmarks]:
        for benchmark in tqdm(benchmarks):
            print(benchmark)
            reward_base = base_performance(env, benchmark)
            reward_greedy = greedy_search(env, benchmark)
            reward_policy = policy_search(env, benchmark)
            df_gflops.loc[len(df_gflops)] = [benchmark, reward_base, reward_greedy, reward_policy]

        df_gflops_list.append(df_gflops)
        
    plot_results(df_gflops_list)




if __name__ == '__main__':
    
    print(args)
    # register_env()
    benchmark = str(args.benchmark)

    with make_env() as env:
        if benchmark in env.datasets.datasets():
            eval_benchmarks(env, benchmark)
        elif benchmark in env.datasets.benchmarks():
            eval_benchmark(env, benchmark)
