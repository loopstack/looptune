# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Union

import numpy as np
from loop_tool_service.models.llvm_autotuning.just_keep_going_env import JustKeepGoingEnv

import compiler_gym
from compiler_gym.envs.compiler_env import CompilerEnv

from compiler_gym.datasets import Benchmark
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import RuntimePointEstimateReward

import loop_tool_service

logger = logging.getLogger(__name__)

_RUNTIME_LOCK = Lock()


class OptimizationTarget(str, Enum):
    CODESIZE = "codesize"
    BINSIZE = "binsize"
    RUNTIME = "runtime"
    FLOPS = "flops_loop_nest"

    @property
    def optimization_space_enum_name(self) -> str:
        return {
            OptimizationTarget.CODESIZE: "IrInstructionCount",
            OptimizationTarget.BINSIZE: "ObjectTextSizeBytes",
            OptimizationTarget.RUNTIME: "Runtime",
            OptimizationTarget.FLOPS: "flops_loop_nest",
        }[self.value]

    def make_env(self, benchmark: Union[str, Benchmark]) -> CompilerEnv:
        # env: LlvmEnv = compiler_gym.make("llvm-v0")
        env: CompilerEnv = loop_tool_service.make("loop_tool-v0")

        # TODO(cummins): This does not work with custom benchmarks, as the URI
        # will not be known to the new environment.
        if str(benchmark).startswith("file:///"):
            benchmark = env.make_benchmark(Path(benchmark[len("file:///") :]))

        env.benchmark = benchmark

        if self.value == OptimizationTarget.CODESIZE:
            env.reward_space = "IrInstructionCountOz"
        elif self.value == OptimizationTarget.BINSIZE:
            env.reward_space = "ObjectTextSizeOz"
        elif self.value == OptimizationTarget.RUNTIME:
            env = RuntimePointEstimateReward(env, warmup_count=0, runtime_count=3)
        elif self.value == OptimizationTarget.FLOPS:
            env.reward_space = "flops_loop_nest"
        else:
            assert False, f"Unknown OptimizationTarget: {self.value}"

        # Wrap the env to ignore errors during search.
        env = JustKeepGoingEnv(env)
        return env

    def final_reward(self, env: CompilerEnv, runtime_count: int = 30) -> float:
        """Compute the final reward of the environment.

        Note that this may modify the environment state. You should call
        :code:`reset()` before continuing to use the environment after this.
        """
        print("*************** FINAL REWARD ******************\n\n")
        # Reapply the environment state in a retry loop.
        actions = list(env.actions)
        env.reset()
        for i in range(1, 5 + 1):
            _, _, done, info = env.multistep(actions)
            if not done:
                break
            logger.warning(
                "Attempt %d to apply actions during final reward failed: %s",
                i,
                info.get("error_details"),
            )
        else:
            raise ValueError("Failed to replay environment's actions")

        if self.value == OptimizationTarget.CODESIZE:
            return env.observation.IrInstructionCountOz() / max(
                env.observation.IrInstructionCount(), 1
            )

        if self.value == OptimizationTarget.BINSIZE:
            return env.observation.ObjectTextSizeOz() / max(
                env.observation.ObjectTextSizeBytes(), 1
            )

        if self.value == OptimizationTarget.RUNTIME:
            with _RUNTIME_LOCK:
                with compiler_gym.make("llvm-v0", benchmark=env.benchmark) as new_env:
                    new_env.reset()
                    new_env.runtime_observation_count = runtime_count
                    new_env.runtime_warmup_count = 0
                    new_env.apply(env.state)
                    final_runtimes = new_env.observation.Runtime()
                    assert len(final_runtimes) == runtime_count

                    new_env.reset()
                    new_env.send_param("llvm.apply_baseline_optimizations", "-O3")
                    o3_runtimes = new_env.observation.Runtime()
                    assert len(o3_runtimes) == runtime_count

                logger.debug("O3 runtimes: %s", o3_runtimes)
                logger.debug("Final runtimes: %s", final_runtimes)
                speedup = np.median(o3_runtimes) / max(np.median(final_runtimes), 1e-12)
                logger.debug("Speedup: %.4f", speedup)

                return speedup
        
        import loop_tool_service
        if self.value == OptimizationTarget.FLOPS:
            with _RUNTIME_LOCK:
                with loop_tool_service.make("loop_tool-v0", benchmark=env.benchmark) as new_env:
                    new_env.reset()
                    print("*************** FINAL REWARD 1******************\n\n")

                    new_env.runtime_observation_count = runtime_count
                    print("*************** FINAL REWARD 2******************\n\n")

                    new_env.runtime_warmup_count = 0
                    print("*************** FINAL REWARD 3******************\n\n")
                    
                    # new_env.apply(env.state)
                    print("*************** FINAL REWARD 4******************\n\n")

                    final_gflops = new_env.observation.flops_loop_nest() / 1e9
                    print("*************** FINAL REWARD 5******************\n\n")

                    # assert len(final_runtimes) == runtime_count
                    print("*************** FINAL REWARD 6******************\n\n")


                    # assert len(o3_runtimes) == runtime_count

                print("*************** FINAL REWARD 9******************\n\n")
                max_gflops = 141.74
                print("Max GFLOPS: %s", max_gflops)
                print("Final GFLOPS: %s", final_gflops)
                speedup = max(np.median(final_gflops) / np.median(max_gflops), 1e-12)
                print("Speedup: %.4f", speedup)

                return speedup

            

        assert False, f"Unknown OptimizationTarget: {self.value}"
