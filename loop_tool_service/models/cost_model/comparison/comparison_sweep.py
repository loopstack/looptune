import os

import loop_tool as lt
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from tqdm import tqdm
# import time
from IPython import display
from ipywidgets import Output


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import loop_tool_service.models.my_net as my_net
import wandb

import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class LoopToolDataset(Dataset):
    def __init__(
        self,
        df,
    ):
        self.df = df

    def __getitem__(self, i):
        j = random.randint(0, len(self.df) - 1)
        # j = (i + 1 )% len(self.df)
        stride_freq_log_0 = np.log2(self.df['stride_tensor'].iloc[i] + 1)
        stride_freq_log_1 = np.log2(self.df['stride_tensor'].iloc[j] + 1)
        x = stride_freq_log_0 - stride_freq_log_1
        y = [ float(self.df['gflops'].iloc[i] > self.df['gflops'].iloc[j]) ]
        return torch.flatten(x.float()).to(device), torch.tensor(y).float().to(device)

    def __len__(self):    
        return len(self.df)



def load_dataset(config):
    from loop_tool_service.paths import LOOP_TOOL_ROOT
    data_path = str(LOOP_TOOL_ROOT) + "/loop_tool_service/models/datasets/tensor_dataset_noanot.pkl"
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
    model = my_net.SmallNetSigmoid(
        in_size=config['size_in'], 
        out_size=config['size_out'], 
        hidden_size=config['hidden_size'],
        num_layers=config['layers'],
        dropout=config['dropout'],
    ).to(device)

    wandb.watch(model, log="all")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. constructed load_model")
    return model


def train_epoch(model, TrainLoader, optimizer, criterion):
    train_losses_batch = []

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. train_epoch")

    for state, cost in TrainLoader:
        model.train()

        # for state, cost in zip(state, cost):
            
        pred_cost = model(state)
        train_loss = criterion(pred_cost, cost)
        # print(state)
        # print(cost, pred_cost)
        train_losses_batch.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    return train_losses_batch


def test_epoch(model, TestLoader, optimizer, criterion):
    test_losses_batch = []

    with torch.no_grad():
        model.eval()
        for state, cost in TestLoader:
            pred_cost =  model(state)
            test_loss = criterion(pred_cost, cost )
            test_losses_batch.append(test_loss.item())
    
    return test_losses_batch


def final_performance(model, TestLoader):
    correct = []

    with torch.no_grad():
        model.eval()

        for state, label in TestLoader:
            for i, (state, label) in enumerate(zip(state, label)):
                
                pred_label = 1 if model(state)[0] > 0.5 else 0

                result =  1 if pred_label == label[0] else 0
                correct.append(result)

    return np.mean(correct)




def train(config=None):
    train_loss = []
    test_loss = []



    with wandb.init(
        project="loop_tool", 
        entity="dejang", 
        config=config
    ):
        config = wandb.config
        trainLoad, testLoad = load_dataset(config)
        model = load_model(config)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.BCELoss(reduction=config['reduction'])

        for epoch in tqdm(range(config['epochs'])):    
            train_losses_batch = train_epoch(model, trainLoad, optimizer, criterion)
            test_losses_batch = test_epoch(model, testLoad, optimizer, criterion)

            wandb.log({
                "train_loss": np.mean(train_losses_batch),
                "test_loss": np.mean(test_losses_batch)}
            )
        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. Final Performance")
        wandb.log({"final_performance": final_performance(model, testLoad)})

    return train_loss, test_loss


if __name__ == "__main__":
 
    sweep_count = 2
    os.environ['WANDB_NOTEBOOK_NAME'] = 'comparison_sweep.ipynb'

    breakpoint()
    sweep_config = {
    "name" : "Comparison-sweep",
    "method": "random",
    "metric": {
        "name": "final_performance",
        "goal": "maximize",
    },
    "parameters" : {
        "hidden_size" : {"values": [ 100, 200, 300 ]},
        "layers" : {"values": [ 2, 3 ]},
        'reduction' : {"values": [ 'sum', 'mean' ]},
        'lr': {
        'distribution': 'log_uniform_values',
        'min': 0.00001,
        'max': 0.1
        },
        "epochs": { "value" : 50 },
        "batch_size": { "value" : 100 },
        "dropout": { "value" : 0.2 },
    }
    }

    sweep_id = wandb.sweep(sweep_config, project="loop_tool")
    wandb.agent(sweep_id=sweep_id, function=train, count=sweep_count)