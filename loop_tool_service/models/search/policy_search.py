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

last_run_path = LOOP_TOOL_ROOT/"loop_tool_service/models/rllib/my_artifacts"




# Training settings
parser = argparse.ArgumentParser(description="LoopTool Optimizer")
parser.add_argument("--policy-model", type=str, help="Path to the RLlib optimized network.")
parser.add_argument("--cost-model", type=str, help="Path to the cost model network.")
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
        # reward_space="runtime",
    )
    # env = compiler_gym.make("loop_tool_env-v0")

    env = TimeLimit(env, max_episode_steps=10)
    return env
    
def load_datasets(env):
    
    # The two datasets we will be using:
    lt_dataset = env.datasets["benchmark://loop_tool_test-v0"]
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

    





if __name__ == '__main__':
    
    print(args)
    # register_env()
    

    with make_env() as env:
        train_benchmarks, val_benchmarks = load_datasets(env)
        i = 0
        for bench in tqdm(train_benchmarks, total=min(len(train_benchmarks), 10)):
            if i == 10: break
            print(bench)
            env.reset(benchmark=bench)

            if args.cost_model != '':
                env.send_param('load_cost_model', args.cost_model)
            if args.policy_model != '':
                env.send_param('load_policy_model', args.policy_model)
            best_actions_reward = json.loads(env.send_param("policy_search", '100, 3'))
            print(best_actions_reward)
            i += 1
            breakpoint()

