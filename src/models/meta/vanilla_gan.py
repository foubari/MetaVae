import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import argparse
from src.models.blocks import *
from tqdm import tqdm



    
class Vanilla_Generator(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers=0, joint_layers=2):
        super(Vanilla_Generator,self).__init__()
        self.latent_dim = latent_dim
        self._system_dim = system_dim
        self._relu = nn.ReLU()
        
        self._hblock = Block(h_dim,h_layers,n_params,n_params)
        self._fcz = nn.Linear(latent_dim,n_params)
        self._joint_block = Block(n_params,joint_layers,2*system_dim,n_params)

    def forward(self,z,h):
        h = self._hblock(h)
        z = self._relu(self._fcz(z))
        z = z+h
        s = self._joint_block(z)
        s = torch.reshape(s,(-1,2,self._system_dim))
        return s

    
    
class Vanilla_Discriminator(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers=2, joint_layers=2):
        super(Vanilla_Discriminator,self).__init__()
        self.latent_dim = latent_dim
        self._sig = nn.Sigmoid()
        self._relu = nn.ReLU()
        
        
        self._hblock = Block(h_dim,h_layers,256,n_params)
        self._fc_x = nn.Linear(system_dim,n_params)
        self._fc_y = nn.Linear(system_dim,n_params)
        self._fc_xy = nn.Linear(2*n_params,256)
        self._fccat = nn.Linear(2*256,n_params)#cat with h
        self._joint_block = Block(n_params,joint_layers,1,n_params)
        
    def forward(self,x,h):
        h = self._hblock(h)
        x,y = x[:,0,:],x[:,1,:]
        x = self._relu(self._fc_x(x))
        y = self._relu(self._fc_y(y))
        x = torch.cat((x,y),1)
        x = self._relu(self._fc_xy(x))
        x = torch.cat((x,h),1)
        x = self._relu(self._fccat(x))
        x = self._joint_block(x)
        x = self._sig(x)
        return x


class Vanilla_GAN(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers_d=2, h_layers_g=0, joint_layers_d=2, joint_layers_g=2):
        super(Vanilla_GAN,self).__init__()
        discriminator_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'system_dim': system_dim,
                          'n_params': n_params, 'h_layers': h_layers_d, 'joint_layers': joint_layers_d}
        generator_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'system_dim': system_dim,
                          'n_params': n_params, 'h_layers': h_layers_g, 'joint_layers': joint_layers_g}
        self.latent_dim = latent_dim
        self.disc = Vanilla_Discriminator(**discriminator_params)
        self.gen = Vanilla_Generator(**generator_params)    
        
    def forward(self, z,h_param):
        return self.gen(z,h_param)
    
    def criterion(self,y,y_hat):
        return nn.BCELoss()(y,y_hat)
    
    def train(self,epochs, train_dataloader, device,  optim_g, optim_d, verbose = False):
        for epoch in tqdm(range(epochs)):
            epoch_g_losses, epoch_d_losses = [],[]
            for data, h_param in iter(train_dataloader):
                self.to(device)
                data, h_param = data.to(device), h_param.to(device)
                noise = torch.randn(data.shape[0],self.latent_dim).to(device)    
                #Training discriminator
                #On real data      
                self.disc.train()
                self.gen.eval()            
                real_output = self.disc(data, h_param)            
                #On fake data
                generation = self(noise,h_param)
                fake_output = self.disc(generation,h_param) 
                real_loss = self.criterion(real_output,torch.ones_like(real_output))
                fake_loss = self.criterion(fake_output,torch.zeros_like(fake_output))
                d_loss = (real_loss.mean() +fake_loss.mean())/2
                optim_d.zero_grad()
                d_loss.backward()
                optim_d.step()

                #Training Generator
                self.disc.eval()
                self.gen.train()
                generation = self(noise,h_param)         
                fake_output = self.disc(generation,h_param) 
                g_loss = self.criterion(fake_output,torch.ones_like(fake_output)).mean()
                optim_g.zero_grad()
                g_loss.backward()
                optim_g.step()
                
                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())
            if verbose:
                epoch_d_loss = np.mean(epoch_d_losses)
                epoch_g_loss = np.mean(epoch_g_losses)
                print('g_loss: ', epoch_g_loss, ' d_loss: ', epoch_d_loss)
                
    @torch.no_grad()
    def generate(self, test_dataloader, unif = .2):
        self.to('cpu')
        data, h = next(iter(test_dataloader))
        latent = torch.rand(h.shape[0],self.latent_dim)*2*unif-unif
        return self(latent,h)
            

    
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Vanilla GAN")

    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of the latent space.')
    parser.add_argument('--h_dim', type=int, default=3, help='Dimension of the hyperparameters.')
    parser.add_argument('--system_dim', type=int, default=180, help='Number of intermediate layers in system block.')
    parser.add_argument('--n_params', type=int, default=64, help='Number of parameters in each layer *.')
    parser.add_argument('--h_layers_d', type=int, default=2, help='Number of intermediate layers in the discriminator hyperparameters block.')
    parser.add_argument('--h_layers_g', type=int, default=2, help='Number of intermediate layers in the generator hyperparameters block.')
    parser.add_argument('--joint_layers_d', type=int, default=3, help='Number of intermediate joint layers in the discriminator.')
    parser.add_argument('--joint_layers_g', type=int, default=3, help='Number of intermediate joint layers in the generator.')

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    vanilla_gan = Vanilla_GAN()