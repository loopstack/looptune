"""This script runs loop_nest and evaluate GFLOPS so we can measure standard deviation.

    $ python loop_tool_service/experiments/eval_loop_nest.py

"""
import logging
import json
import random

from compiler_gym.util.logging import init_logging
from compiler_gym.util.registration import register

import loop_tool_service
from loop_tool_service.service_py.datasets import mm128_128_128
from loop_tool_service.service_py.rewards import  flops_loop_nest_reward



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
                mm128_128_128.Dataset(),
            ],
        },
    )

register_env()

def main():
    # Use debug verbosity to print out extra logging information.
    init_logging(level=logging.CRITICAL)
    register_env()

    walks = 10
    flops = {}
    with loop_tool_service.make_env("loop_tool_env-v0") as env:
        for bench in env.datasets["benchmark://mm128_128_128-v0"]:
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{bench}")
            env.reset(benchmark=bench)

            for i in range(walks):
                available_actions = json.loads(env.send_param("available_actions", ""))
                action = random.choice(available_actions)
                env.step(action=env.action_space.from_string(action))
                agent_str = env.send_param("print_looptree", "") 
                if agent_str in flops:
                    flops[agent_str].append(env.observation['flops_loop_nest'])
                else:
                    flops[agent_str] = [env.observation['flops_loop_nest']]

        

        for k,v in flops.items(): print(k, sorted(v))    
                

if __name__ == "__main__":
    main()
