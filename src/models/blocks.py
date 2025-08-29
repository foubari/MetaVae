import torch
import torch.nn as nn
import argparse


class Block(nn.Module):
    def __init__(self,input_size,n_layers,output_size,n_params):
        super(Block,self).__init__()
        layers = [] 
        layers.append(nn.Linear(input_size,n_params))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_params,output_size))
        self._layers =  nn.Sequential(*layers)
    def forward(self,x):
        return self._layers(x)
    
class UBlock(nn.Module):
    # Block that takes as input the point cloud dataset, the x,y coordinates are fed to self._fc_x,resp self._fc_y
    def __init__(self,input_size,n_layers,output_size,n_params):
        super(UBlock,self).__init__()
        self._relu = nn.ReLU()
        self._fc_x = nn.Linear(input_size,n_params)
        self._fc_y = nn.Linear(input_size,n_params)
        layers = [] 
        layers.append(nn.Linear(2*n_params,n_params))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_params,output_size))
        self._layers =  nn.Sequential(*layers)
    def forward(self,x):
        x, y = x[:,0,:],x[:,1,:]
        x = self._relu(self._fc_x(x))
        y = self._relu(self._fc_y (y))
        x = torch.cat((x,y),1)
        return self._layers(x)
    
    
    
class LBlock(nn.Module):
    # Block that takes as input the point cloud dataset, the x,y coordinates are fed to self._fc_x,resp self._fc_y
    def __init__(self,input_size,n_layers,output_size,n_params):
        super(LBlock,self).__init__()
        layers = [] 
        layers.append(nn.Linear(input_size,n_params))
        layers.append(nn.ReLU())
        for n in range(n_layers):
            layers.append(nn.Linear(n_params, n_params))
            layers.append(nn.ReLU())
        self._layers =  nn.Sequential(*layers)
        self._fcx = nn.Linear(n_params,output_size)
        self._fcy = nn.Linear(n_params,output_size)
    def forward(self,x):
        x = self._layers(x)
        y = self._fcy(x)
        x = self._fcx(x)
        return x,y