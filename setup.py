#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import distutils.util
import setuptools
import os

version = "0.2.3"

with open("requirements.txt") as f:
    requirements = [ln.split("#")[0].rstrip() for ln in f.readlines()]

setuptools.setup(
    name="looptune_service",
    version=version,
    description="Example code for CompilerGym",
    author="Facebook AI Research",
    url="https://github.com/facebookresearch/CompilerGym",
    license="MIT",
    install_requires=requirements,
    packages=[
        "looptune_service",
        "looptune_service.benchmarks",
        "looptune_service.experiments",
        "looptune_service.experiments.compare_searches",
        "looptune_service.experiments.compare_tvm",
        "looptune_service.experiments.eval_loop_nest",
        "looptune_service.experiments.eval_loop_tool",
        "looptune_service.service_py",
        "looptune_service.service_py.datasets",
        "looptune_service.service_py.env",
        "looptune_service.service_py.rewards",        
        "looptune_service.models",
        "looptune_service.models.rllib",        
        "looptune_service.models.rllib.config",
        "looptune_service.models.rllib.config.a3c",
        "looptune_service.models.rllib.config.dqn",
        "looptune_service.models.rllib.config.impala",
        "looptune_service.models.rllib.config.ppo",
        "looptune_service.models.rllib.config.apex_dqn",
    ],
    python_requires=">=3.8",
    platforms=[distutils.util.get_platform()],
    zip_safe=False,
)

# vi3: Install by using this command.
# python -m pip install .
# According to this: https://stackoverflow.com/questions/66125129/unknownextra-error-when-installing-via-setup-py-but-not-via-pip


###############################################################
# Set up root directory
###############################################################
print("\nSet root directory: \n\nexport LOOPTUNE_ROOT=%s"%os.getcwd())
# print(f"export MAX_GFLOPS={os.popen('/private/home/dejang/tools/loop_tool/extern/loop_nest/build/apps/gflops.avx2.fp32').read()}")
