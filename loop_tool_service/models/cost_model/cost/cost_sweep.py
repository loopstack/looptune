import argparse
import os
import time

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from pathlib import Path

from tqdm import tqdm

import loop_tool_service.models.my_net as my_net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import wandb
from loop_tool_service.paths import LOOP_TOOL_ROOT


import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LoopToolDataset(Dataset):
    def __init__(
        self,
        df,
    ):
        self.df = df

    def __getitem__(self, i):
        stride_freq_log_0 = np.log2(self.df['stride_tensor'].iloc[i] + 1)
        label = self.df['gflops'].iloc[i]
        return torch.flatten(stride_freq_log_0.float()).to(device), torch.tensor(label).float().to(device)

    def __len__(self):    
        return len(self.df)


def load_dataset(config):
    from loop_tool_service.paths import LOOP_TOOL_ROOT
    data_path = str(LOOP_TOOL_ROOT) + "/loop_tool_service/benchmarks/observations_db/mm8_16_8_16_8_16_db.pkl"
    if config['data_size'] < 0:
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_pickle(data_path).iloc[:config['data_size'], :]

    loop_tool_dataset = LoopToolDataset(df=df)

    test_size = len(loop_tool_dataset.df) // 5
    train_size = len(loop_tool_dataset.df) - test_size

    print(f'Dataset training validation = {train_size}, {test_size}')
    train_set, test_set = torch.utils.data.random_split(loop_tool_dataset, [train_size, test_size])

    trainLoad = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    testLoad = DataLoader(test_set, batch_size=config['batch_size'], shuffle=True)

    config['size_in'] = len(torch.flatten(df['stride_tensor'].iloc[0]))
    config['size_out'] = 1
    
    return trainLoad, testLoad


def load_model(config):
    model = my_net.SmallNet(
        in_size=config['size_in'], 
        out_size=config['size_out'], 
        hidden_size=config['hidden_size'],
        num_layers=config['layers'],
        dropout=config['dropout'],
    ).to(device)

    # wandb.watch(model, log="all") # <<<<<<<< this breaks torch.jit.script
    return model


def train_epoch(model, TrainLoader, optimizer, criterion):
    train_losses_batch = []

    for state, cost in TrainLoader:
        model.train()

        # for state, cost in zip(state, cost):
            
        pred_cost = torch.flatten(model(state)) # good
        # pred_cost = model(state) # bad

        train_loss = criterion(pred_cost, cost)
        # print(state)
        # print(cost, pred_cost)
        train_losses_batch.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        wandb.log({"batch loss": train_loss.item()})

    return train_losses_batch


def test_epoch(model, TestLoader, optimizer, criterion):
    test_losses_batch = []

    with torch.no_grad():
        model.eval()
        for state, cost in TestLoader:
            # for state, cost in zip(state, cost):

            pred_cost =  torch.flatten(model(state))
            # pred_cost =  model(state)
            test_loss = criterion(pred_cost, cost )
            test_losses_batch.append(test_loss.item())
    
    return test_losses_batch


def final_performance_compare(model, TestLoader):
    correct = []

    with torch.no_grad():
        model.eval()

        for state, cost in TestLoader:
            for i, (state, cost) in enumerate(zip(state,cost)):
                if i == 0: 
                    prev_state, prev_cost = state, cost
                    continue

                prev_pred_cost = model(prev_state)
                pred_cost = model(state)
                result =  1 if (prev_pred_cost > pred_cost) == (prev_cost > cost) else 0
                correct.append(result)

                prev_state, prev_cost = state, cost

    return np.mean(correct)

def final_performance(model, TestLoader, loss):
    losses = []

    with torch.no_grad():
        model.eval()

        for state, cost in TestLoader:
            losses.append(float(loss(model(state), cost)))

    return np.mean(losses)


def save_model(model, model_name):
    model_path = LOOP_TOOL_ROOT/f'loop_tool_service/models/weights/{model_name}'
    model_path.parent.mkdir(exist_ok=True, parents=True)
    # torch.save(model.state_dict(), model_path)
    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save(str(model_path)) # Save

    wandb.run.save(str(model_path))


def train(config=None):
    train_loss = []
    test_loss = []

    with wandb.init(
        project="loop_stack_cost_model", 
        entity="dejang",
    ):
        config = wandb.config
        wandb.run.name = f"cost_{wandb.config['layers']}_{wandb.config['hidden_size']}"
        trainLoad, testLoad = load_dataset(config)
        model = load_model(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.MSELoss()

        for epoch in tqdm(range(config['epochs'])):    
            train_losses_batch = train_epoch(model, trainLoad, optimizer, criterion)
            test_losses_batch = test_epoch(model, testLoad, optimizer, criterion)

            wandb.log({
                    "train_loss": np.mean(train_losses_batch),
                    "test_loss": np.mean(test_losses_batch)
                }
            )
        fp = final_performance(model, testLoad, torch.nn.L1Loss())
        print(f"final_performance {fp}" )
        wandb.log({"final_performance": fp })
        save_model(model=model, model_name='cost_model.pt')


    return train_loss, test_loss

def update_default_config(sweep_config=None):
    for key, val in default_config.items():
        if key in sweep_config:
            if type(val) == dict:
                val.update(sweep_config[key])
            else:
                default_config[key] = sweep_config[key]

    return default_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sweep",  type=int, nargs='?', const=2, default=1, help="Run with wandb sweeps"
    )
    args = parser.parse_args()


    default_config = {
        "name" : "Cost-sweep",
        "method": "random",
        "metric": {
            "name": "final_performance",
            "goal": "maximize",
        },
        "parameters" : {
            "layers" : { "value": 8},
            "hidden_size" : { "value": 500},
            'lr': { "value": 1e-6},
            "epochs": { "value" : 10000 },
            "batch_size": { "value" : 50 },
            "dropout": { "value" : 0.2 }, # dropout cannot be 0 for some reason
            "data_size": { "value" : 10 },
        }
    }
    os.environ['WANDB_NOTEBOOK_NAME'] = 'cost_sweep.ipynb'

    sweep_config = {
        "name" : "Cost-sweep",
        "method": "grid",
        "metric": {
            "name": "final_performance",
            "goal": "maximize",
        },
        "parameters":{
            "layers" : {"values": [ 4, 8, 20]},
            "hidden_size" : {"values": [ 100, 500, 1000]},
            # 'lr': {
            # 'distribution': 'log_uniform_values',
            # 'min': 0.000001,
            # 'max': 0.001
            # },
            # "epochs": { "value" : 5000 },
            # "batch_size": { "value" : 50 },
            # "dropout": { "values" : [0, 0.2] },
            "data_size": { "value" : 10 },
        }
    }


    default_config = update_default_config(sweep_config)
    sweep_id = wandb.sweep(default_config, project="loop_stack_cost_model")

    wandb.agent(sweep_id=sweep_id, function=train, count=args.sweep)