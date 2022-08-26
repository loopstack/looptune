import os
from pathlib import Path

import loop_tool as lt
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from tqdm import tqdm
import loop_tool_service.models.my_net as my_net
from loop_tool_service.paths import LOOP_TOOL_ROOT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import wandb

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
  "hidden_size" : 300,
  "layers": 10,
  'lr': 3.8e-4,
  "epochs": int(1e0),
  "batch_size": 500,
  "dropout": 0.2,
}


class LoopToolDataset(Dataset):
    def __init__(
        self,
        df,
    ):
        self.df = df

    def __getitem__(self, i):
        stride_freq_log_0 = np.log2(self.df['program_tensor'].iloc[i] + 1)
        label = self.df['gflops'].iloc[i]
        return torch.flatten(stride_freq_log_0.float()).to(device), torch.tensor([label]).float().to(device)

    def __len__(self):    
        return len(self.df)



def load_dataset(config):
    from loop_tool_service.paths import LOOP_TOOL_ROOT
    data_path = str(LOOP_TOOL_ROOT) + "/loop_tool_service/models/datasets/tensor_dataset_noanot.pkl"
    df = pd.read_pickle(data_path).iloc[:, :]
    loop_tool_dataset = LoopToolDataset(df=df)

    test_size = len(loop_tool_dataset.df) // 5
    train_size = len(loop_tool_dataset.df) - test_size

    print(f'Dataset training validation = {train_size}, {test_size}')
    train_set, test_set = torch.utils.data.random_split(loop_tool_dataset, [train_size, test_size])

    trainLoad = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    testLoad = DataLoader(test_set, batch_size=config['batch_size'], shuffle=True)

    config['size_in'] = len(torch.flatten(df['program_tensor'].iloc[0]))
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

    return model



def train_epoch(model, TrainLoader, optimizer, criterion):
    train_losses_batch = []

    for state, cost in TrainLoader:
        model.train()

        # for state, cost in zip(state, cost):
            
        pred_cost = torch.flatten(model(state)) # good
        pred_cost = model(state) # bad
        # breakpoint()
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
            # for state, cost in zip(state, cost):

            # pred_cost =  torch.flatten(model(state))
            pred_cost =  model(state)
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


def train(config, model, trainLoad, testLoad):
    train_loss = []
    test_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss(reduction='mean')

    for epoch in tqdm(range(config['epochs'])):    
        train_losses_batch = train_epoch(model, trainLoad, optimizer, criterion)
        test_losses_batch = test_epoch(model, testLoad, optimizer, criterion)

        train_loss.append(np.mean(train_losses_batch))
        test_loss.append(np.mean(test_losses_batch))
    
        plt.title('Loss (blue-train, red-test)')
        plt.plot(train_loss, color='blue')
        plt.plot(test_loss, color='red')

        plt.tight_layout()
        plt.savefig('tmp.png')
        
    print(f'Final performance = {final_performance(model, testLoad)}')
            
    return train_loss, test_loss


config = sweep_config
trainLoad, testLoad = load_dataset(config=config)
model = load_model(config)

train_loss, test_loss = train(config, model, trainLoad, testLoad)

model_path = Path(os.path.dirname(os.path.realpath(__file__))+'/my_artifacts/cost.pt')
model_path.parent.mkdir(exist_ok=True, parents=True)

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save(str(model_path)) # Save
os.symlink(model_path, LOOP_TOOL_ROOT/'loop_tool_service/models/weights/cost.pt')


from loop_tool_service.paths import LOOP_TOOL_ROOT
data_path = str(LOOP_TOOL_ROOT) + "/loop_tool_service/models/datasets/tensor_dataset_noanot.pkl"

df = pd.read_pickle(data_path)
diff = []
model.eval()
for index, row in df.iterrows():
    # print(row['program_tensor'])
    stride_freq_log_0 = np.log2(df['program_tensor'].iloc[index] + 1)
    
    state = torch.flatten(stride_freq_log_0).float().to(device)
    label = row['gflops']
    
    pred = model(state).item()
    diff.append(abs(pred - label))
    print(pred, label)