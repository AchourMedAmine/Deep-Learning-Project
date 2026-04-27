import torch 
from torch.utils.data import Dataset
import numpy as np 
from transformers import AutoFeatureExtractor

class BreathDataset(Dataset):
    def __init__(self,X,y,devices_id,processor,train=True):
        self.X=X
        self.y=y
        self.devices_id=devices_id
        self.processor=processor
        self.train=train
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        wav=self.X[idx]
        if self.train :
            if np.random.random() <0.5 : 
                wav=wav*np.random.uniform(0.9,1.1)
                wav=wav+np.random.normal(0,0.0001,wav.shape)
        inputs=self.processor(wav,sampling_rate=16000,return_tensors="pt")
        input_values=inputs.input_values.squeeze(0)
        return input_values,torch.tensor(self.y[idx],dtype=torch.long),self.device_ids[idx]

