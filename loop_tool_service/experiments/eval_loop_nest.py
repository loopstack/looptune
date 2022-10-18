"""This script runs loop_nest and evaluate GFLOPS so we can measure standard deviation.

    $ python loop_tool_service/experiments/eval_loop_nest.py

    as a result for each benchmark we get row {mean_gflops, std_gflops, mean_time, std_time}
    To visualize copy results to:
    https://docs.google.com/spreadsheets/d/1GQXZT0rVbj9pAa2Y6tAki3a7T3CHzP4zWdOV9qiPgZU/edit#gid=0
"""
import logging
import time
import random

from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register

import loop_tool_service
from loop_tool_service.service_py.datasets import mm128_128_128, mm32_8_16_8_4_16
from loop_tool_service.service_py.rewards import  flops_loop_nest_reward
from statistics import mean, stdev


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
                # mm128_128_128.Dataset(),
                mm32_8_16_8_4_16.Dataset(),
            ],
        },
    )


def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    repeat = 10
    flops = {}
    with loop_tool_service.make_env("loop_tool_env-v0") as env:
        for i, bench in enumerate(env.datasets["benchmark://mm32_8_16_8_4_16-v0"]):
            if i == 20: break
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{bench}")

            for _ in range(repeat):
                env.reset(benchmark=bench)

                # available_actions = json.loads(env.send_param("available_actions", ""))
                # action = random.choice(available_actions)
                # env.step(action=env.action_space.from_string(action))

                start = time.time()
                gflops = env.observation['flops_loop_nest']
                mtime = time.time() - start

                agent_str = env.send_param("print_looptree", "") 
                print(gflops)
                if agent_str in flops:
                    flops[agent_str]['gflops'].append(gflops)
                    flops[agent_str]['time'].append(mtime)
                else:
                    flops[agent_str] = {'gflops': [gflops], 'time': [mtime]}

        
        stat = []
        for item in flops.values(): 
            stat.append([mean(item['gflops']), stdev(item['gflops']), mean(item['time']), stdev(item['time'])])   

        print('gflops [mean stdev], time [mean stdev]')
        for x in sorted(stat):
            print(','.join( [ str(item) for item in x] ))    

if __name__ == "__main__":
    main()
