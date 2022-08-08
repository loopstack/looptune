import os

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from tqdm import tqdm

from loop_tool_service.models.slurm import SubmititJobSubmitter
import loop_tool_service.models.my_net as my_net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from ray import tune
from copy import deepcopy
import wandb
import subprocess

import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'

slurm_time_min = 10
sweep_count = 2
sweep_config = {
    'lr': tune.uniform(1e-3, 1e-6),
    "gamma": tune.uniform(0.5, 0.99),
    "horizon": tune.choice([None, 5, 20]),
}



from loop_tool_service.models.rllib.rllib_torch_sweep import train, default_config, stop_criteria

def submit_job():
    config = deepcopy(default_config)    
    for key, value in sweep_config.items():
        config[key] = value.sample()

    executor = SubmititJobSubmitter(
        timeout_min=slurm_time_min
    ).get_executor()

    job = executor.submit(train, config, stop_criteria)
    print(f"Job ID: {job.job_id}")


if __name__ == "__main__":
    for _ in range(sweep_count):
        submit_job()
