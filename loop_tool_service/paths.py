from pathlib import Path
import os
import subprocess
from typing import Iterable
import logging

import subprocess
from datetime import datetime

'''
Purpose:
    In this file we define all paths important for the project.
    To run the project user needs to set LOOPTUNE_ROOT to the path of loop_tool_env directory.
    BENCHMARKS_PATH - path to user-defined benchmarks source-code
    LOOP_TOOL_SERVICE_PY - path to loop_tool_env backend service 
'''

LOOPTUNE_ROOT = Path(os.environ.get("LOOPTUNE_ROOT"))
assert LOOPTUNE_ROOT, "\n\nInitialize envvar LOOPTUNE_ROOT to path of the loop_tool_env folder \n"


BENCHMARKS_PATH: Path = Path(
    LOOPTUNE_ROOT / "loop_tool_service/benchmarks"
)


LOOP_TOOL_SERVICE_PY: Path = Path(
    LOOPTUNE_ROOT / "loop_tool_service/service_py/example_service.py"
)

logging.info(f"What is the path {LOOP_TOOL_SERVICE_PY}")
logging.info(f"Is that file: {LOOP_TOOL_SERVICE_PY.is_file()}")
assert LOOP_TOOL_SERVICE_PY.is_file(), "Service script not found"


def create_log_dir(experiment_name):
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_dir = "/".join([str(LOOPTUNE_ROOT), "results", experiment_name, timestamp])
    os.makedirs(log_dir)

    return log_dir