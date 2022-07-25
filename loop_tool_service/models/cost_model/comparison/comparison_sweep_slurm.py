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

from comparison_sweep import train

sweep_count = 10
sweep_config = {
  "name" : "Comparison-sweep",
  "method": "random",
  "metric": {
    "name": "final_performance",
    "goal": "maximize",
  },
  "parameters" : {
    "hidden_size" : {"values": [ 100, 200, 300 ]},
    "layers" : {"values": [ 3, 4, 5 ]},
    'reduction' : {"values": [ 'sum', 'mean' ]},
    'lr': {
      'distribution': 'log_uniform_values',
      'min': 0.00001,
      'max': 0.1
    },
    "epochs": { "value" : 1000 },
    "batch_size": { "value" : 100 },
    "dropout": { "value" : 0.2 },
    "data_size": { "value" : -1 },
    "timeout_min": { "value": 200}
  }
}



def submit_job():
        executor = SubmititJobSubmitter(
            timeout_min=sweep_config['parameters']['timeout_min']['value']
        ).get_executor()

        job = executor.submit(train)
        print(f"Job ID: {job.job_id}")


if __name__ == "__main__":

    sweep_id = wandb.sweep(sweep_config, project="loop_tool")
    wandb.agent(sweep_id=sweep_id, function=submit_job, count=sweep_count)