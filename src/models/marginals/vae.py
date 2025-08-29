import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import argparse
from src.models.blocks import *
from tqdm import tqdm
    

class Encoder(nn.Module):
    def __init__(self, n_layers, n_params, latent_dim, data_size):
        super(Encoder,self).__init__()    
        self.latent_dim = latent_dim
        self._layers =  UBlock(data_size,n_layers,2*latent_dim, n_params)# Changer en Block
    def forward(self,x):
        x = self._layers(x)
        mu,sigma = x[:,:self.latent_dim],x[:,self.latent_dim:]
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, n_layers, n_params, latent_dim, data_size):
        super(Decoder,self).__init__()    
        self._output_size  = data_size
        self._layers = LBlock(latent_dim,n_layers,data_size,n_params)
    def forward(self,z):
        x_out, y_out = self._layers(z)
        return  torch.reshape(torch.cat((x_out,y_out),1),(-1,2,self._output_size))
    
    
# class VAE(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self._encoder = Encoder(args.n_layers_e, args.n_params, args.latent_dim, args.data_size)
#         self._decoder = Decoder(args.n_layers_d, args.n_params, args.latent_dim, args.data_size)
#         self._latent_dim = args.latent_dim
#         self._output_size = args.data_size

#     def sample(self,mu,sigma):
#         div = torch.device('cpu') if mu.get_device() else torch.device('cuda:0')
#         Normal_distrib = torch.distributions.Normal(0, 1)
#         Normal_distrib.loc = Normal_distrib.loc.to(div) 
#         Normal_distrib.scale = Normal_distrib.scale.to(div)
#         return mu + sigma*Normal_distrib.sample(mu.shape)

#     def forward(self,x):
#         mu,sigma = self._encoder(x)
#         latent = self.sample(mu,sigma)
#         reconstruction = self._decoder(latent)
#         return mu, sigma, reconstruction 

class VAE(nn.Module):
    def __init__(self,n_layers_e = 4, n_layers_d = 3 , n_params = 12, latent_dim = 2, data_size = 60): # Paper values , for density change data_size = 30 and latent_dim = 1
        super().__init__()
        self._encoder = Encoder(n_layers_e, n_params, latent_dim, data_size)
        self._decoder = Decoder(n_layers_d, n_params, latent_dim, data_size)
        self._latent_dim = latent_dim
        self._output_size = data_size

    def sample(self, mu, sigma):
        device_index = mu.get_device()
        if device_index == -1:
            div = torch.device('cpu')
        else:
            div = torch.device(f'cuda:{device_index}')

        Normal_distrib = torch.distributions.Normal(0, 1)
        Normal_distrib.loc = Normal_distrib.loc.to(div)
        Normal_distrib.scale = Normal_distrib.scale.to(div)
        return mu + sigma * Normal_distrib.sample(mu.shape)


    def forward(self,x):
        mu,sigma = self._encoder(x)
        latent = self.sample(mu,sigma)
        reconstruction = self._decoder(latent)
        return mu, sigma, reconstruction
    
    def criterion(self,mu,sigma,reconstruction,data):  
        recon_loss = nn.MSELoss()(reconstruction,data)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + sigma- mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    @torch.no_grad()
    def val(self,val_dataloader):
        self.to('cpu')
        losses=[]
        for data in iter(val_dataloader):
            mu,sigma,recon = self(data)
            all_losses = self.criterion(mu,sigma,recon,data)
            loss,_,_ = all_losses
            losses.append(loss)
        return np.mean(losses)

    def train(self, epochs, train_dataloader, device, optimizer,scheduler=None,early_stopper=None,val_dataloader=None, verbose = False):
        for epoch in tqdm(range(epochs)):
            self.to(device)
            epoch_losses=[]
            for data in iter(train_dataloader):
                data = data.to(device)
                mu,sigma,recon = self(data)
                all_losses = self.criterion(mu,sigma,recon,data)
                loss,_,_ = all_losses
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)
            if verbose:
                print('loss ',epoch_loss)
            if scheduler:
                scheduler.step(np.mean(epoch_losses)) 
            if early_stopper:
                val_score = self.val(val_dataloader)
                if early_stopper.early_stop(val_score): 
                    print('Early stopping')
                    break
                    
    @torch.no_grad()
    def test(self,test_dataloader):
        self.to('cpu')
        data = next(iter(test_dataloader))
        mu,sigma,recon = self(data)
        all_losses = self.criterion(mu,sigma,recon,data)
        loss,_,_ = all_losses
        return loss
    
    def generate(self,latent):
        return self._decoder(latent)
    
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Variational Autoencoder")

    parser.add_argument('--n_layers_e', type=int, default=5,
                        help='Number of layers in the encoder.')
    parser.add_argument('--n_layers_d', type=int, default=3,
                        help='Number of layers in the decoder.')
    parser.add_argument('--n_params', type=int, default=12,
                        help='Number of parameters in each layer.')
    parser.add_argument('--latent_dim', type=int, default=2,
                        help='Dimension of the latent space.')
    parser.add_argument('--data_size', type=int, default=60,
                        help='Size of the input data.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # vae = VAE(args)

    vae = VAE() 

    print(vae)