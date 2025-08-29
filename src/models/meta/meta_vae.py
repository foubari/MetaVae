import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import torch
import torch.nn as nn
import argparse
from src.models.blocks import *
from tqdm import tqdm
    
    
    
class Meta_Encoder(nn.Module):
    def __init__(self,latent_dim=4,h_dim=3,density_dim = 30, nested_cylinders_dim = 120,n_params = 64, h_layers = 2, d_layers = 1,  joint_layers = 3):
        super(Meta_Encoder,self).__init__()
        self.latent_dim = latent_dim
        self._hblock = Block(h_dim,h_layers,256,n_params) # Condition block à retirer
        self._int_density_block = UBlock(density_dim,d_layers,256,n_params) # équivalent time series
        self._ext_density_block = UBlock(density_dim,d_layers,256,n_params) # équivalent time series
        
        self._fc_x = nn.Linear(nested_cylinders_dim,n_params) # à retirer
        self._fc_y = nn.Linear(nested_cylinders_dim,n_params) # à retirer
        self._fc_xy = nn.Linear(2*n_params,256)# à retirer 
        
        self._joint_block = Block(256*4,joint_layers,2*latent_dim,n_params)
        
        self._relu = nn.ReLU()
        
    def forward(self,x,h,di,de):
        
        h = self._hblock(h)
        di = self._int_density_block(di)
        de =  self._ext_density_block(de)

        x,y = x[:,0,:],x[:,1,:]# à retirer

        x = self._relu(self._fc_x(x))# à retirer
        y = self._relu(self._fc_y(y))# à retirer
        
        x = torch.cat((x,y),1)# à retirer
        x = self._relu(self._fc_xy(x))# à retirer
        x = torch.cat((x,h,di,de),1)
        
        z = self._joint_block(x)
        return z[:,self.latent_dim:],z[:,:self.latent_dim] # mu, logvar

    
    
class Meta_Decoder(nn.Module):
    def __init__(self,latent_dim=4,h_dim=3,n_params = 64, density_latent_dim = 1, cylinders_latent_dim =2,h_layers = 0,d_layers=3,c_layers=4):
        super(Meta_Decoder,self).__init__()
        self.latent_dim = latent_dim
        self._cylinders_latent_dim = cylinders_latent_dim
        self._zlayer =  nn.Linear(latent_dim,n_params)
        self._hblock =  Block(h_dim,h_layers,n_params,n_params)
        self._int_density_block = Block(n_params,d_layers,density_latent_dim,n_params)
        self._ext_density_block = Block(n_params,d_layers,density_latent_dim,n_params)
        self._cylinders_block = Block(n_params,c_layers,2*cylinders_latent_dim,n_params)   
        
        self._relu = nn.ReLU()
    def forward(self,z,h):
        h = self._hblock(h)# à retirer 
        z = self._relu(self._zlayer(z))# à retirer 
        z = z+h# à retirer 
        di = self._int_density_block(z)
        de = self._ext_density_block(z)
        c = self._cylinders_block(z)     
        ci =  c[:,:self._cylinders_latent_dim]
        ce = c[:,self._cylinders_latent_dim:]
        return ci,ce,di,de
    

    
    

        
class Meta_VAE(nn.Module):
    def __init__(self, latent_dim=4, h_dim=3, density_dim=30, nested_cylinders_dim=120, n_params=64,
                 he_layers=2, de_layers=1, joint_layers=3, density_latent_dim=1, cylinders_latent_dim=2,
                 hd_layers=0, dd_layers=3, cd_layers=4):
        super(Meta_VAE, self).__init__()
        self.latent_dim = latent_dim
        encoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'density_dim': density_dim,
                          'nested_cylinders_dim': nested_cylinders_dim, 'n_params': n_params,
                          'h_layers': he_layers, 'd_layers': de_layers, 'joint_layers': joint_layers}
        decoder_params = {'latent_dim': latent_dim, 'h_dim': h_dim, 'n_params': n_params, 'density_latent_dim': density_latent_dim,
                          'cylinders_latent_dim': cylinders_latent_dim, 'h_layers': hd_layers, 'd_layers': dd_layers, 'c_layers': cd_layers}
        self._encoder = Meta_Encoder(**encoder_params)
        self._decoder = Meta_Decoder(**decoder_params)

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
        z_int, z_ext, zdi,zde = self._decoder(latent,h_param) # h_param à retirer
        return mu,sigma, z_int, z_ext, zdi,zde
    
    def criterion(self,mu,sigma,c_gen,c,di_gen,de_gen,d_int,d_ext):
        c_recon_loss = nn.MSELoss()(c_gen,c) # à retirer 
        d_recon_loss = nn.MSELoss()(di_gen,d_int)+nn.MSELoss()(de_gen,d_ext)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + sigma- mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        return c_recon_loss+ d_recon_loss + kl_loss, c_recon_loss, d_recon_loss, kl_loss
    
    @torch.no_grad()
    def val(self,density_vae,cylinder_vae,val_dataloader):
        self.to('cpu')
        cylinder_vae.to('cpu')
        density_vae.to('cpu')
        losses=[]
        for c, h_param, d_int,d_ext in iter(val_dataloader):
            mu,sigma,z_int, z_ext, zdi, zde = self(c, h_param, d_int, d_ext)
            int_gen = cylinder_vae._decoder(z_int)
            ext_gen =  cylinder_vae._decoder(z_ext) 
            di_gen = density_vae._decoder(zdi)
            de_gen = density_vae._decoder(zde)
            c_gen = torch.cat((int_gen,ext_gen),2)
            all_losses = self.criterion(mu,sigma,c_gen,c,di_gen,de_gen,d_int,d_ext)
            loss,_,_,_ = all_losses
            losses.append(loss)
        return np.mean(losses)
                   
    def train(self,epochs,train_dataloader,density_vae,cylinder_vae,device,optimizer,scheduler=None,early_stopper=None,val_dataloader=None,verbose=False):
        losses, kl_losses, c_r_losses, d_r_losses = [],[],[],[]
        means,stds = [],[]
        for epoch in tqdm(range(epochs)):
            self.to(device)
            density_vae.to(device)
            cylinder_vae.to(device)
            ip=0
            epoch_losses,kl_epoch_losses,c_r_epoch_losses,d_r_epoch_losses = [],[],[],[]
            for c, h_param, d_int,d_ext in iter(train_dataloader):
                c = c.to(device)
                h_param = h_param.to(device)
                d_int = d_int.to(device)
                d_ext = d_ext.to(device)

                mu,sigma,z_int, z_ext, zdi, zde = self(c, h_param, d_int, d_ext)
                int_gen = cylinder_vae._decoder(z_int)
                ext_gen =  cylinder_vae._decoder(z_ext) 
                di_gen = density_vae._decoder(zdi)
                de_gen = density_vae._decoder(zde)
                c_gen = torch.cat((int_gen,ext_gen),2) # àretirer
                all_losses = self.criterion(mu,sigma,c_gen,c,di_gen,de_gen,d_int,d_ext)
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
                print(losses[-1])
            if scheduler:
                scheduler.step(losses[-1])
            if early_stopper:
                val_score = self.val(density_vae,cylinder_vae,val_dataloader)
                if early_stopper.early_stop(val_score): 
                    print('Early stopping')
                    break
            
#         return losses, kl_losses, c_r_losses,d_r_losses
    
    @torch.no_grad()
    def test(self,density_vae,cylinder_vae,test_dataloader):
        self.to('cpu')
        cylinder_vae.to('cpu')
        density_vae.to('cpu')
        c, h_param, d_int,d_ext = next(iter(test_dataloader))
        mu,sigma,z_int, z_ext, zdi, zde = self(c, h_param, d_int, d_ext)
        int_gen = cylinder_vae._decoder(z_int)
        ext_gen =  cylinder_vae._decoder(z_ext) 
        di_gen = density_vae._decoder(zdi)
        de_gen = density_vae._decoder(zde)
        c_gen = torch.cat((int_gen,ext_gen),2)
        all_losses = self.criterion(mu,sigma,c_gen,c,di_gen,de_gen,d_int,d_ext)
        loss,_,_,_ = all_losses
        return loss
                                          
    @torch.no_grad()
    def generate(self, density_vae,cylinder_vae,test_dataloader, unif = .2):
        self.to('cpu')
        _, h, _, _ = next(iter(test_dataloader))#à retirer
        latent = torch.rand(h.shape[0],self.latent_dim)*2*unif-unif
        # latent = torch.randn(h.shape[0],self.latent_dim) sample from N(0,I)
        z_int, z_ext, zdi,zde = self._decoder(latent,h)
        return cylinder_vae(z_int) , cylinder_vae(z_ext), density_vae(zdi), density_vae(zde)
    
    
