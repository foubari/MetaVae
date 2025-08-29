import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import argparse
from src.models.blocks import *
from tqdm import tqdm

    

class Vanilla_Encoder(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers=2, joint_layers=2):
        super(Vanilla_Encoder,self).__init__()
        self.latent_dim = latent_dim
        self._hblock = Block(h_dim,h_layers,256,n_params)
        self._fc_x = nn.Linear(system_dim,n_params)
        self._fc_y = nn.Linear(system_dim,n_params)
        self._fc_xy = nn.Linear(2*n_params,256)
        self._fccat = nn.Linear(2*256,n_params)#cat with h
        self._joint_block = Block(n_params,joint_layers,2*latent_dim,n_params)
        self._relu = nn.ReLU()
    def forward(self,x,h):
        h = self._hblock(h)
        x,y = x[:,0,:],x[:,1,:]
        x = self._relu(self._fc_x(x))
        y = self._relu(self._fc_y(y))
        x = torch.cat((x,y),1)
        x = self._relu(self._fc_xy(x))
        x = torch.cat((x,h),1)
        x = self._relu(self._fccat(x))
        z = self._joint_block(x)
        return z[:,self.latent_dim:],z[:,:self.latent_dim]
    
    
    
class Vanilla_Decoder(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers=0, joint_layers=2):
        super(Vanilla_Decoder,self).__init__()
        self.latent_dim = latent_dim
        self._system_dim = system_dim
        self._relu = nn.ReLU()#nn.ReLU()
        
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
    
    
class Vanilla_VAE(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, system_dim=180, n_params=64, h_layers_e=2, h_layers_d=0, joint_layers_e=2, joint_layers_d=2):
        super(Vanilla_VAE, self).__init__()

        encoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'system_dim': system_dim,
                          'n_params': n_params, 'h_layers': h_layers_e, 'joint_layers': joint_layers_e}
        decoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'system_dim': system_dim,
                          'n_params': n_params, 'h_layers': h_layers_d, 'joint_layers': joint_layers_d}
        self.latent_dim = latent_dim
        self._encoder = Vanilla_Encoder(**encoder_params)
        self._decoder = Vanilla_Decoder(**decoder_params) 
        
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
    
    def forward(self,data, h_param):
        mu,sigma = self._encoder(data, h_param)
        latent = self.sample(mu,sigma)
        recon = self._decoder(latent,h_param)
        return mu,sigma, recon
    
    
    
    
    def criterion(self,mu,sigma,reconstruction,data):  
        recon_loss = nn.MSELoss()(reconstruction,data)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + sigma- mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    @torch.no_grad()
    def val(self,val_dataloader):
        self.to('cpu')
        losses=[]
        for data, h in iter(val_dataloader):
            mu,sigma,recon = self(data, h)
            all_losses = self.criterion(mu,sigma,recon,data)
            loss,_,_ = all_losses
            losses.append(loss)
        return np.mean(losses)
                          
    def train(self, epochs, train_dataloader, device, optimizer,scheduler=None,early_stopper=None,val_dataloader=None, verbose = False):
        for epoch in tqdm(range(epochs)):
            self.to(device)
            epoch_losses=[]
            for data,h in iter(train_dataloader):
                data,h = data.to(device),h.to(device)
                mu,sigma,recon = self(data,h)
                all_losses = self.criterion(mu,sigma,recon,data)
                loss,_,_ = all_losses
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            epoch_loss = np.mean(epoch_losses)
            if verbose:
                print(epoch_loss)
            if scheduler:
                scheduler.step(epoch_loss) 
            if early_stopper:
                val_score = self.val(val_dataloader)
                if early_stopper.early_stop(val_score): 
                    print('Early stopping')
                    break
                    
    @torch.no_grad()
    def test(self,test_dataloader):
        self.to('cpu')
        data, h = next(iter(test_dataloader))
        mu,sigma,recon = self(data, h)
        all_losses = self.criterion(mu,sigma,recon,data)
        loss,_,_ = all_losses
        return loss
    
    @torch.no_grad()
    def generate(self, test_dataloader, unif = .2):
        self.to('cpu')
        data, h = next(iter(test_dataloader))
        latent = torch.rand(h.shape[0],self.latent_dim)*2*unif-unif
        return self._decoder(latent,h)
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Vanilla VAE")

    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of the latent space.')
    parser.add_argument('--h_dim', type=int, default=3, help='Dimension of the hyperparameters.')
    parser.add_argument('--system_dim', type=int, default=180, help='Number of intermediate layers in system block.')
    parser.add_argument('--n_params', type=int, default=64, help='Number of parameters in each layer *.')
    parser.add_argument('--h_layers_e', type=int, default=2, help='Number of intermediate layers in the encoder hyperparameters block.')
    parser.add_argument('--h_layers_d', type=int, default=2, help='Number of intermediate layers in the decoder hyperparameters block.')
    parser.add_argument('--joint_layers_e', type=int, default=3, help='Number of intermediate joint layers in the encoder.')
    parser.add_argument('--joint_layers_d', type=int, default=3, help='Number of intermediate joint layers in the decoder.')

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    vanilla_vae = Vanilla_VAE(latent_dim=args.latent_dim, h_dim=args.h_dim, system_dim=args.system_dim, n_params=args.n_params, 
                           h_layers_e=args.h_layers_e, h_layers_d=args.h_layers_d, joint_layers_e=args.joint_layers_e, joint_layers_d=args.joint_layers_d)