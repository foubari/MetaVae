import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse

sys.path.append(os.path.abspath('..'))

# data_dir = "../dataset/marginals"
data_dir  =  os.path.join(os.path.abspath('..'),"src","dataset","marginals")

class My_Dataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        return data

class DataModule():
    def __init__(self, data_dir=data_dir, batch_size=1024, is_cylinder=True): #is_cylinder --> serie temporelle nom
        super().__init__()
        self._data_dir = data_dir
        self.dl_dict = {'batch_size': batch_size, 'shuffle': True}
        self.is_cylinder = is_cylinder
        if self.is_cylinder:
            self.unit_comp = "cylinders"
            self.data_size = 60
        else:
            self.unit_comp = "densities"
            self.data_size = 30

    def to_tensor(self, arr):
        tensor_data = torch.tensor(arr)
        return torch.cat((torch.unsqueeze(tensor_data[:, :, 0], 1), torch.unsqueeze(tensor_data[:, :, 1], 1)), 1).float()

    def setup(self):
        with open(os.path.join(self._data_dir,self.unit_comp, 'train', self.unit_comp + '.npy'), 'rb') as f:
            data_train = np.load(f)
        self._train_set = My_Dataset(self.to_tensor(data_train))
        
        with open(os.path.join(self._data_dir,self.unit_comp, 'val', self.unit_comp + '.npy'), 'rb') as f:
            data_val = np.load(f)
        self._val_set = My_Dataset(self.to_tensor(data_val))

        with open(os.path.join(self._data_dir,self.unit_comp, 'test', self.unit_comp + '.npy'), 'rb') as f:
            data_test = np.load(f)
        self._test_set = My_Dataset(self.to_tensor(data_test))

    def train_dataloader(self):
        return DataLoader(self._train_set, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self._val_set, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=5000)


    
    


def parse_arguments():
    parser = argparse.ArgumentParser(description='DataModule arguments')
    parser.add_argument('--data_dir', type=str, default='../dataset/meta',
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for the DataLoader')
    parser.add_argument('--is_cylinder', action='store_true',
                        help='Whether to use cylinders (default) or densities')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    data_module = DataModule(data_dir=args.data_dir, batch_size=args.batch_size, is_cylinder=args.is_cylinder)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()