import pandas as pd
import numpy as np
import torch
device = 'cpu'

df = pd.read_pickle("tensor_dataset_noanot.pkl") 
df.head()
breakpoint()

for i in range(len(df)):
    stride_freq_log_0 = np.log2(df['stride_tensor'].iloc[i] + 1)
    label = df['gflops'].iloc[i]

    print(torch.flatten(stride_freq_log_0.float()).to(device), torch.tensor(label).float().to(device)) 