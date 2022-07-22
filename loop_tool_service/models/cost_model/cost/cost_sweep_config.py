import os

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from tqdm import tqdm

import loop_tool_service.models.my_net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import wandb

import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'




hyperparameter_defaults = dict(
    layers = 4,
    hidden_size = 300,
    lr = 1e-4,
    dropout = 0.2,
    epochs = 100,
    batch_size = 100,
    data_size = 200
)

wandb.init(config=hyperparameter_defaults, project="loop_tool_env")
config = wandb.config




class LoopToolDataset(Dataset):
    def __init__(
        self,
        df,
    ):
        self.df = df

    def __getitem__(self, i):
        stride_freq_log_0 = np.log2(self.df['program_tensor'].iloc[i] + 1)
        label = self.df['gflops'].iloc[i]
        return torch.flatten(stride_freq_log_0.float()).to(device), torch.tensor(label).float().to(device)

    def __len__(self):    
        return len(self.df)


def load_dataset(config):

    from loop_tool_service.paths import LOOP_TOOL_ROOT
    data_path = str(LOOP_TOOL_ROOT) + "/loop_tool_service/models/datasets/tensor_dataset_noanot.pkl"

    if config.data_size < 0:
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_pickle(data_path).iloc[:config.data_size, :]

    loop_tool_dataset = LoopToolDataset(df=df)

    test_size = len(loop_tool_dataset.df) // 5
    train_size = len(loop_tool_dataset.df) - test_size

    print(f'Dataset training validation = {train_size}, {test_size}')
    train_set, test_set = torch.utils.data.random_split(loop_tool_dataset, [train_size, test_size])

    trainLoad = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    testLoad = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)

    config.size_in = len(torch.flatten(df['program_tensor'].iloc[0]))
    config.size_out = 1
    
    return trainLoad, testLoad


def load_model(config):
    model_path = "model_weights.pt"
    model = my_net.SmallNet(
        in_size=config.size_in, 
        out_size=config.size_out, 
        hidden_size=config.hidden_size,
        num_layers=config.layers,
        dropout=config.dropout,
    ).to(device)

    wandb.watch(model, log="all")

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


def final_performance(model, TestLoader):
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



def train():
    train_loss = []
    test_loss = []

    trainLoad, testLoad = load_dataset(config)
    model = load_model(config)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(config.epochs)):    
        train_losses_batch = train_epoch(model, trainLoad, optimizer, criterion)
        test_losses_batch = test_epoch(model, testLoad, optimizer, criterion)

        wandb.log({
                "train_loss": np.mean(train_losses_batch),
                "test_loss": np.mean(test_losses_batch)
            }
        )
    pf = final_performance(model, testLoad)
    print(f"final_performance {pf}" )
    wandb.log({"final_performance": final_performance(model, testLoad) })
    return train_loss, test_loss




if __name__ == '__main__':
   train()