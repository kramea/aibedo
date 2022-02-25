import numpy as np
from typing import List, Dict, Union, Tuple



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#import xarray




#parameters
key_list = ['in_highres', 'out_highres', 'in_lowres', 'out_lowres']



class TemporalDS(Dataset):
    def __init__(self, data_dict, hr_seq_len, lr_seq_len, hlkup, dlkup):
        self.datadict = data_dict
        self.hr_seq_len = hr_seq_len
        self.lr_seq_len = lr_seq_len
        self.freq_factor = 24.0
        self.hlookup = hlkup
        self.dlookup = dlkup
        self.num_samples = self.hlookup.shape[0] #hardcoded right now for this example DS
        
        
    def __getitem__(self, index)-> Dict[str, torch.Tensor]:
        hr_idx = self.hlookup[index]
        lr_idx = self.dlookup[index]
        
        
        sample = {}
        
        #populate for high res first (i.e. hourly)
        sample[f'x_d_1H'] = self.datadict['in_highres'][hr_idx - self.hr_seq_len + 1: hr_idx+1]
        sample[f'y_1H'] = self.datadict['out_highres'][hr_idx - self.hr_seq_len + 1: hr_idx+1]
        
        sample[f'x_d_1D'] = self.datadict['in_lowres'][lr_idx - self.lr_seq_len + 1: lr_idx+1]
        sample[f'y_1D'] = self.datadict['out_lowres'][lr_idx - self.lr_seq_len + 1: lr_idx+1]

        return sample

    def __len__(self):
        return self.num_samples

def load_rainfall_data(in_HR_fname, in_LR_fname, out_HR_fname, out_LR_fname):
    x_d_1H_array = torch.tensor(np.load(in_HR_fname))
    x_d_1D_array = torch.tensor(np.load(in_LR_fname))
    y_1H_array = torch.tensor(np.load(out_HR_fname))
    y_1D_array = torch.tensor(np.load(out_LR_fname))
    
    print("in_HR: ", x_d_1H_array.shape, x_d_1H_array.dtype)
    print("in_LR: ",x_d_1D_array.shape, x_d_1D_array.dtype)
    print("out_HR: ", y_1H_array.shape, y_1H_array.dtype)
    print("out_LR: ",y_1D_array.shape, y_1D_array.dtype)
    
    data_dict = dict(zip(key_list,[x_d_1H_array, y_1H_array, x_d_1D_array,  y_1D_array]))
    
    return data_dict
    
    
    
def load_lookuptable(HR_lookup_fname, LR_lookup_fname):
    """Metadata specifically for rainfall data"""
    H_lookup = np.load(HR_lookup_fname)
    D_lookup = np.load(LR_lookup_fname)
    
    return H_lookup, D_lookup