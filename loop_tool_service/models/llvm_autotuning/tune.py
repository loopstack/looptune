# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys

import hydra
from loop_tool_service.models.llvm_autotuning.experiment import Experiment
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from compiler_gym.util.shell_format import indent
from compiler_gym.util.registration import register



import loop_tool_service
from loop_tool_service.service_py.datasets import loop_tool_dataset
from loop_tool_service.service_py.rewards import flops_loop_nest_reward

import loop_tool_service.models.qAgentsDict as q_agents

def register_env():
    register(
        id="loop_tool-v0",
        entry_point="compiler_gym.service.client_service_compiler_env:ClientServiceCompilerEnv",
        kwargs={
            "service": loop_tool_service.paths.LOOP_TOOL_SERVICE_PY,
            "rewards": [
                flops_loop_nest_reward.RewardScalar(),
                ],
            "datasets": [
                loop_tool_dataset.Dataset(),
            ],
        },
    )


@hydra.main(config_path="config", config_name="default")
def main(config: DictConfig) -> None:
    
    # Parse the config to pydantic models.
    OmegaConf.set_readonly(config, True)
    try:
        model: Experiment = Experiment(working_directory=os.getcwd(), **config)
    except ValidationError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    register_env()
    print("Experiment configuration:")
    print()
    print(indent(model.yaml()))
    print()
    model.run()
    print()
    print("Results written to", model.working_directory)


if __name__ == "__main__":
    main()
