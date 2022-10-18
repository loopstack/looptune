"""
    This script takes generated benchmark and create file of input observations and GFLOPS.

    $ python loop_tool_service/benchmarks/labelator.py --benchmark=mm32_8_16_8_4_16

"""
import argparse
import logging
import pandas as pd

from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register

import loop_tool_service
from loop_tool_service.service_py.rewards import  flops_loop_nest_reward
from statistics import mean, stdev


from loop_tool_service.paths import BENCHMARKS_PATH
import importlib


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bench", type=str, help="Benchmark to generate", required=True
)

def register_env(bench):
    bench_module = importlib.import_module(f"loop_tool_service.service_py.datasets.{bench}")
    register(
        id="loop_tool_env-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardTensor(),
                ],
            "datasets": [
                bench_module.Dataset(),
            ],
        },
    )
    return bench_module.Dataset().name


def main():
    args = parser.parse_args()
    save_path = f'{BENCHMARKS_PATH}/observations_db/{args.bench}_db.pkl'

    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    dataset_url = register_env(args.bench)
    df = pd.DataFrame(columns=['agent', 'loops_tensor', 'stride_tensor', 'gflops'])
    
    with loop_tool_service.make_env("loop_tool_env-v0") as env:
        for i, bench in enumerate(env.datasets[dataset_url]):
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{bench}")

            env.reset(benchmark=bench)
            df.loc[i] = [
                env.send_param("print_looptree", ""),
                env.observation['loops_tensor'],
                env.observation['stride_tensor'],
                env.observation['flops_loop_nest'],
            ]
            
    
    df.to_pickle(save_path)

if __name__ == "__main__":
    main()
