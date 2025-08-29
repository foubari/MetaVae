import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import argparse
from src.models.blocks import *
from tqdm import tqdm




class SMEncoder(nn.Module):
    def __init__(self,latent_dim=4,h_dim=3,density_dim = 30, nested_cylinders_dim = 120,n_params = 64, h_layers = 2, d_layers = 1,  joint_layers = 3):
        super(SMEncoder,self).__init__()
        self.latent_dim = latent_dim
        self._hblock = Block(h_dim,h_layers,256,n_params)
        self._int_density_block = UBlock(density_dim,d_layers,256,n_params)
        self._ext_density_block = UBlock(density_dim,d_layers,256,n_params)
        
        self._fc_x = nn.Linear(nested_cylinders_dim,n_params)
        self._fc_y = nn.Linear(nested_cylinders_dim,n_params)
        self._fc_xy = nn.Linear(2*n_params,256)
        
        self._joint_block = Block(256*4,joint_layers,2*latent_dim,n_params)
        
        self._relu = nn.ReLU()
        
    def forward(self,x,h,di,de):
        
        h = self._hblock(h)
        di = self._int_density_block(di)
        de =  self._ext_density_block(de)

        x,y = x[:,0,:],x[:,1,:]

        x = self._relu(self._fc_x(x))
        y = self._relu(self._fc_y(y))
        
        x = torch.cat((x,y),1)
        x = self._relu(self._fc_xy(x))
        x = torch.cat((x,h,di,de),1)
        
        z = self._joint_block(x)
        return z[:,self.latent_dim:],z[:,:self.latent_dim]

    
    
class SMDecoder(nn.Module):
    def __init__(self,latent_dim=4,h_dim=3,density_dim = 30, cylinders_dim = 60,n_params = 64,h_layers = 0,d_layers=3,c_layers=3):
        super(SMDecoder,self).__init__()
        self.latent_dim = latent_dim
        self._cylinders_dim = cylinders_dim
        self._density_dim = density_dim
        self._zlayer =  nn.Linear(latent_dim,n_params)
        self._hblock =  Block(h_dim,h_layers,n_params,n_params)
        self._int_density_block = Block(n_params,d_layers,2*density_dim,n_params)
        self._ext_density_block = Block(n_params,d_layers,2*density_dim,n_params)
        self._cylinders_block = Block(n_params,c_layers,4*cylinders_dim,n_params)   
        
        self._relu = nn.ReLU()
    def forward(self,z,h):
        # print('begin decoder')
        # print('h,z',h.shape,z.shape)
        h = self._hblock(h)
        z = self._relu(self._zlayer(z))
        z = z+h
        di = self._int_density_block(z)
        de = self._ext_density_block(z)
        c = self._cylinders_block(z)  

        
        di = torch.reshape(di,(-1,2,self._density_dim))
        de = torch.reshape(de,(-1,2,self._density_dim))
        ci,ce = c[:,:2*self._cylinders_dim],c[:,2*self._cylinders_dim:]
        ci = torch.reshape(ci,(-1,2,self._cylinders_dim))
        ce = torch.reshape(ce,(-1,2,self._cylinders_dim))
        # print('di',di.shape, 'ci',ci.shape)
        # print('end decoder')
        return ci,ce,di,de
    
    
    
    
'''uncomment for terminal use'''
# class SMVAE(nn.Module):
#     def __init__(self):
#         super(SMVAE,self).__init__()
#         self.latent_dim = args.latent_dim
#         encoder_params = {'latent_dim': args.latent_dim, 'h_dim': args.h_dim, 'density_dim': args.density_dim,
#                           'nested_cylinders_dim': args.nested_cylinders_dim, 'n_params': args.n_params,
#                           'h_layers': args.he_layers, 'd_layers': args.de_layers, 'joint_layers': args.joint_layers}
#         decoder_params = {'latent_dim': args.latent_dim, 'h_dim': args.h_dim, 'n_params': args.n_params, 'density_dim': args.density_dim,
#                           'cylinders_latent_dim': args.cylinders_latent_dim, 'h_layers': args.hd_layers, 'd_layers': args.dd_layers, 'c_layers': args.cd_layers}
#         self._encoder = SMEncoder(**encoder_params)
#         self._decoder = SMDecoder(**decoder_params)
        
#         def sample(self,mu,sigma):
#         div = torch.device('cpu') if mu.get_device() else torch.device('cuda:0')
#         Normal_distrib = torch.distributions.Normal(0, 1)
#         Normal_distrib.loc = Normal_distrib.loc.to(div) 
#         Normal_distrib.scale = Normal_distrib.scale.to(div)
#         return mu + sigma*Normal_distrib.sample(mu.shape)
    
#     def forward(self,data, h_param, d_int,d_ext):
#         mu,sigma = self._encoder(data, h_param, d_int,d_ext)
#         latent = self.sample(mu,sigma)
#         ci,ce,di,de = self._decoder(latent,h_param)
#         return mu,sigma, ci,ce,di,de
        
        
class SMVAE(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, density_dim=30, nested_cylinders_dim=120, n_params=64, he_layers=2, de_layers=1, joint_layers=3, hd_layers=0, dd_layers=3, cd_layers=4):
        super(SMVAE, self).__init__()
        self.latent_dim = latent_dim
        encoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'density_dim': density_dim,
                          'nested_cylinders_dim': nested_cylinders_dim, 'n_params': n_params,
                          'h_layers': he_layers, 'd_layers': de_layers, 'joint_layers': joint_layers}
        decoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'n_params': n_params, 'density_dim': density_dim,
                          'cylinders_dim': nested_cylinders_dim//2, 'h_layers': hd_layers, 'd_layers': dd_layers, 'c_layers': cd_layers}
        self._encoder = SMEncoder(**encoder_params)
        self._decoder = SMDecoder(**decoder_params)

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
    
    def forward(self,c, h_param, d_int,d_ext):
        mu,sigma = self._encoder(c, h_param, d_int,d_ext)
        latent = self.sample(mu,sigma)
        ci,ce,di,de = self._decoder(latent,h_param)
        # print('forward',mu.shape,h_param.shape,'ci',ci.shape,'di',di.shape,'latent',latent.shape)
        return mu,sigma, ci,ce,di,de
    
    def criterion(self,mu,sigma,c_gen, di_gen, de_gen, c, d_int, d_ext):
        # print('begin criterion')
        # print('c',c_gen.shape,c.shape)
        c_recon_loss = nn.MSELoss()(c_gen,c)
        d_recon_loss = nn.MSELoss()(di_gen,d_int)+nn.MSELoss()(de_gen,d_ext)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + sigma- mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        # print('end criterion')
        return c_recon_loss+ d_recon_loss+ + kl_loss, c_recon_loss, d_recon_loss, kl_loss
    
    @torch.no_grad()
    def val(self,val_dataloader):
        self.to('cpu')
        losses=[]
        for c, h_param, d_int,d_ext in iter(val_dataloader):
            mu,sigma, ci,ce,di_gen,de_gen = self(c, h_param, d_int, d_ext)
            c_gen = torch.cat((ci,ce),2)
            all_losses = self.criterion(mu,sigma,c_gen, di_gen, de_gen, c, d_int, d_ext)
            loss,_,_,_ = all_losses
            losses.append(loss)
        return np.mean(losses)
                   
    def train(self,epochs, train_dataloader, device, optimizer,scheduler=None,early_stopper=None,val_dataloader=None, verbose = False):
        losses, kl_losses, c_r_losses, d_r_losses = [],[],[],[]
        means,stds = [],[]
        for epoch in tqdm(range(epochs)):
            self.to(device)
            epoch_losses,kl_epoch_losses,c_r_epoch_losses,d_r_epoch_losses = [],[],[],[]
            for c, h_param, d_int,d_ext in iter(train_dataloader):
                c = c.to(device)
                h_param = h_param.to(device)
                d_int = d_int.to(device)
                d_ext = d_ext.to(device)

                mu,sigma, ci,ce,di_gen,de_gen = self(c, h_param, d_int, d_ext)
                # print('h',h_param.shape)
                # print('ci',ci.shape,ce.shape)
                # print('di',di_gen.shape,d_int.shape)
                c_gen = torch.cat((ci,ce),2)
                # print('c',c_gen.shape,c.shape)
                all_losses = self.criterion(mu,sigma,c_gen, di_gen, de_gen, c, d_int, d_ext)
                loss,c_r_loss, d_r_loss, kl_loss = all_losses

                epoch_losses.append(loss.item())
                c_r_epoch_losses.append(c_r_loss.item())
                d_r_epoch_losses.append(d_r_loss.item())
                kl_epoch_losses.append(kl_loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            losses.append(np.mean(epoch_losses))
            kl_losses.append(np.mean(kl_epoch_losses))
            c_r_losses.append(np.mean(c_r_epoch_losses))
            d_r_losses.append(np.mean(d_r_epoch_losses))
            if verbose:
                print('loss: ',losses[-1])
            if scheduler:
                scheduler.step(losses[-1])
            if early_stopper:
                val_score = self.val(val_dataloader)
                if early_stopper.early_stop(val_score): 
                    print('Early stopping')
                    break
            
        # return losses, kl_losses, c_r_losses,d_r_losses
    
    @torch.no_grad()
    def test(self,test_dataloader):
        self.to('cpu')
        c, h_param, d_int,d_ext = next(iter(test_dataloader))
        mu,sigma, ci,ce,di_gen,de_gen = self(c, h_param, d_int, d_ext)
        c_gen = torch.cat((ci,ce),2)
        all_losses = self.criterion(mu,sigma,c_gen, di_gen, de_gen, c, d_int, d_ext)
        loss,_,_,_ = all_losses
        return loss
                                          
    @torch.no_grad()
    def generate(self, test_dataloader, unif = .2):
        self.to('cpu')
        _, h, _, _ = next(iter(test_dataloader))
        latent = torch.rand(h.shape[0],self.latent_dim)*2*unif-unif
        return self._decoder(latent,h)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Simplified Meta-VAE")

    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of the latent space.')
    parser.add_argument('--h_dim', type=int, default=3, help='Dimension of the hyperparameters space.')
    parser.add_argument('--density_dim', type=int, default=30, help='Dimension of the density space.')
    parser.add_argument('--n_params', type=int, default=64, help='Number of parameters in each layer.')
    parser.add_argument('--he_layers', type=int, default=2, help='Number of layers in the encoder for h.')
    parser.add_argument('--de_layers', type=int, default=1, help='Number of layers in the encoder for density.')
    parser.add_argument('--joint_layers', type=int, default=3, help='Number of joint layers in the encoder.')
    parser.add_argument('--density_dim', type=int, default=30, help='Dimension of the densities.')
    parser.add_argument('--nested_cylinders_dim', type=int, default=120, help='Dimension of the nested cylinders.')
    parser.add_argument('--hd_layers', type=int, default=0, help='Number of layers in the decoder for h.')
    parser.add_argument('--dd_layers', type=int, default=3, help='Number of layers in the decoder for density.')
    parser.add_argument('--cd_layers', type=int, default=4, help='Number of layers in the decoder for cylinders.')

    return parser.parse_args()


if __name__ == "__main__":
    # args = parse_arguments()
    # smvae = SMVAE(args)
    smvae = SMVAE()