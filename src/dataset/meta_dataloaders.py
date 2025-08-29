import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# sys.path.append(os.path.abspath('..'))

data_dir  =  os.path.join(os.path.abspath('..'),"src","dataset","meta")

class Meta_Dataset(Dataset):
    def __init__(self,dataset,h_param,d_int,d_ext,whole_system = False):
        self.whole_system = whole_system
        self._dataset = dataset.float() # cylindres à retirer
        self._h_param = h_param.float()
        self._d_int = d_int.float() # densitées
        self._d_ext = d_ext.float()
    def __len__(self):
        return len(self._h_param)
    def __getitem__(self,idx):
        data = self._dataset[idx]
        h_param = self._h_param[idx] # à retirer
        d_int = self._d_int[idx]
        d_ext = self._d_ext[idx]
        if self.whole_system: # àretirer
            data = torch.cat((data,d_int,d_ext),1)
            return data,h_param
        return data, h_param, d_int, d_ext
    
    
class DataModule():
    def __init__(self, data_dir = data_dir, batch_size = 1024,whole_system = False):
        super().__init__() 
        self.whole_system = whole_system #à retier
        self.batch_size = batch_size
        self._data_dir = data_dir
        self._batch_size = batch_size 
        self.dl_dict = {'batch_size': self.batch_size, 'shuffle': True}
        

    def to_tensor(self, arr):
        tensor_data = torch.tensor(arr)
        return torch.cat((torch.unsqueeze(tensor_data[:,:,0],1),torch.unsqueeze(tensor_data[:,:,1],1)),1).float() # à vérifier

    def get_hyperparams(self, path): # à retirer
        dataset = pd.read_csv(os.path.join(path,'dataset'))
        xs = np.array(dataset.x)
        ys = np.array(dataset.y)
        masses = np.array(dataset.m_cube)
        masses = torch.unsqueeze(torch.tensor(masses),1)
        xs = torch.unsqueeze(torch.tensor(xs),1)
        ys = torch.unsqueeze(torch.tensor(ys),1)
        return torch.cat((masses,xs,ys),1)

    def get_cylinders(self,path):
        int_path  = os.path.join(path,'int_cylinder.npy')
        ext_path  = os.path.join(path,'ext_cylinder.npy')

        with open(int_path, 'rb') as f:
            int_cylinder = np.load(f)
        with open(ext_path, 'rb') as f:
            ext_cylinder = np.load(f)
            
        ext_tensor_cylinders, int_tensor_cylinders =  self.to_tensor(ext_cylinder), self.to_tensor(int_cylinder)
        return torch.cat(( int_tensor_cylinders, ext_tensor_cylinders),2)

    def get_circular_densities(self,path):
        int_path = os.path.join(path,'int_density.npy')
        ext_path = os.path.join(path,'ext_density.npy')
        
        with open(int_path, 'rb') as f:
            int_density = np.load(f)
        with open(ext_path, 'rb') as f:
            ext_density = np.load(f)
            
        return self.to_tensor(int_density), self.to_tensor(ext_density)


        
    def setup(self):
        
        # get hyperparameters
        train_dir = os.path.join(self._data_dir,'train')
        val_dir = os.path.join(self._data_dir,'val')
        test_dir = os.path.join(self._data_dir,'test')
        exp_dir = os.path.join(self._data_dir,'experiment')

        # get hyperparameters
        train_hp = self.get_hyperparams(train_dir)
        val_hp = self.get_hyperparams(val_dir)  
        test_hp = self.get_hyperparams(test_dir)
        exp_hp = self.get_hyperparams(exp_dir)

        # get cylinders
        train_cylinders = self.get_cylinders(train_dir)
        val_cylinders = self.get_cylinders(val_dir)  
        test_cylinders = self.get_cylinders(test_dir)
        exp_cylinders = self.get_cylinders(exp_dir)

        # get densities
        int_train_densities , ext_train_densities = self.get_circular_densities(train_dir)
        int_val_densities , ext_val_densities = self.get_circular_densities(val_dir)  
        int_test_densities, ext_test_densities = self.get_circular_densities(test_dir)
        int_exp_densities, ext_exp_densities = self.get_circular_densities(exp_dir)

        # train/val sets
        self._train_set = Meta_Dataset(train_cylinders, train_hp, int_train_densities, ext_train_densities,self.whole_system)   
        self._val_set = Meta_Dataset(val_cylinders, val_hp, int_val_densities, ext_val_densities,self.whole_system) 
        self._test_set = Meta_Dataset(test_cylinders, test_hp, int_test_densities, ext_test_densities,self.whole_system)
        self._exp_set = Meta_Dataset(exp_cylinders, exp_hp, int_exp_densities, ext_exp_densities,self.whole_system)

        

    def train_dataloader(self):
        return DataLoader(self._train_set ,  **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self._val_set, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self._test_set, **self.dl_dict)
    
    def exp_dataloader(self):
        return DataLoader(self._exp_set, batch_size = 20000)
    
    
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='DataModule arguments')
    parser.add_argument('--data_dir', type=str, default='../dataset/meta',
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for the DataLoader')
    parser.add_argument('--whole_system', type=bool, default=False,
                        help='Whether to use the whole_system flag')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    data_module = DataModule(data_dir=args.data_dir, batch_size=args.batch_size, whole_system=args.whole_system)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    exp_loader = exp_module.exp_dataloader()