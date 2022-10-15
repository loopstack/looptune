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


import wandb
import subprocess

import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'


sweep_count = 1
sweep_config = {
    "name" : "Cost-sweep",
    "method": "random",
    "metric": {
        "name": "final_performance",
        "goal": "maximize",
    },
    "parameters" : {
        "hidden_size" : {"values": [ 30, 300 ]},
        "layers" : {"values": [ 5, 10]},
        'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-7,
        'max': 1e-5
        },
        "epochs": { "value" : 10000 },
        "batch_size": { "value" : 1000 },
        "dropout": { "value" : 0.2 },
        "data_size": { "value" : 100000 },
        "timeout_min": { "value": 180}
    }
}



from loop_tool_service.models.cost_model.cost.cost_sweep import train

def submit_job():
        executor = SubmititJobSubmitter(
            timeout_min=sweep_config['parameters']['timeout_min']['value']
        ).get_executor()

        job = executor.submit(train)
        print(f"Job ID: {job.job_id}")


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project="loop_stack_cost_model")
    wandb.agent(sweep_id=sweep_id, function=submit_job, count=sweep_count)