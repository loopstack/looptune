'''
In this experiment we evaluate environmeant with cost/policy model on
several searches and plot:
    - executable GFLOPS 
    - compile_time
    - executable actions
'''

import argparse

import compiler_gym
from compiler_gym.wrappers import TimeLimit

import loop_tool_service
from loop_tool_service.models.evaluator import Evaluator
from loop_tool_service.paths import LOOP_TOOL_ROOT
import shutil

from loop_tool_service.paths import LOOP_TOOL_ROOT

from os.path import exists
import wandb


'''
To download network from wandb say: dejang/loop_stack_cost_model/k4ztid39
'''
weights_path = LOOP_TOOL_ROOT/"loop_tool_service/models/weights"
experiment_path = LOOP_TOOL_ROOT/"loop_tool_service/experiments"


# Training settings
parser = argparse.ArgumentParser(description="LoopTool Optimizer")
parser.add_argument("--policy", type=str, nargs='?', const=f"{weights_path}/policy.pt", default='', help="Path to the RLlib optimized network.")
parser.add_argument("--cost", type=str, nargs='?', const=f"{weights_path}/cost.pt", default='', help="Path to the cost model network.")
parser.add_argument("--benchmark", type=str, nargs='?', const='benchmark://loop_tool_test-v0/mm_127x127x127_36_40_20', default='benchmark://loop_tool_test-v0', help="Benchmark to run the search")
parser.add_argument("--size", type=int, nargs='?', default=10, help="Size of benchmarks to evaluate")
parser.add_argument("--steps", type=int, default=10, help="Length of sequence of actions to evaluate")

args = parser.parse_args()
cost_path, policy_path = None, None


def make_env() -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment."""
    
    env = loop_tool_service.make(
        "loop_tool_env-v0",
        observation_space="loops_tensor",
        reward_space="flops_loop_nest_tensor",
    )
    env = TimeLimit(env, max_episode_steps=10)
    return env
    

def resolve_policy(policy_path):
    if policy_path == '' or exists(policy_path):
        return policy_path
    try:
        wandb.restore('policy_model.pt', run_path=policy_path)
    except:
        print('Policy not found')
        exit(1)
        
    shutil.move("policy_model.pt", weights_path/'policy.pt')
    return weights_path/'policy.pt'



def resolve_cost(cost_path):
    if cost_path == '' or exists(cost_path):
        return cost_path
    try:
        wandb.restore('cost_model.pt', run_path=cost_path)
    except:
        print('Cost path not found')
        exit(1)
        
    shutil.move("cost_model.pt", weights_path/'cost.pt')
    return weights_path/'cost.pt'


if __name__ == '__main__':
    print(args)
    policy_path = resolve_policy(args.policy)
    cost_path = resolve_cost(args.cost)

    evaluator = Evaluator(steps=args.steps, cost_path=cost_path, policy_path=policy_path)

    with make_env() as env:
        benchmark = str(args.benchmark)
        if benchmark in env.datasets.datasets():
            benchmarks = env.datasets[benchmark].benchmarks()
        elif benchmark in env.datasets.benchmarks():
            benchmarks = [benchmark]
        else:
            print(f"Benchmark: {benchmark} doesn't exist")
            exit()
        
        # evaluator.evaluate(env, benchmarks, {'bruteforce_ln': evaluator.searches['bruteforce_ln']})
        evaluator.evaluate(env, benchmarks, evaluator.searches)
        evaluator.save(path=experiment_path)
