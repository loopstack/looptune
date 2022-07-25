import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers, dropout=0):
        super(SmallNet,self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_size
            layer_out = hidden_size
            if i == 0:
                layer_in = in_size
            elif i == num_layers - 1:
                layer_out = out_size

            self.layers.append(nn.Linear(layer_in,layer_out))
            torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        for layer in self.layers:
            x = self.dropout(F.leaky_relu(layer(x)))

        return x



class SmallNetSigmoid(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, num_layers, dropout=0):
        super(SmallNetSigmoid,self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_size
            layer_out = hidden_size
            if i == 0:
                layer_in = in_size
            elif i == num_layers - 1:
                layer_out = out_size

            self.layers.append(nn.Linear(layer_in,layer_out))
            torch.nn.init.xavier_uniform_(self.layers[i].weight)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        for layer in self.layers:
            x = self.dropout(F.leaky_relu(layer(x)))

        return self.sigmoid(x)