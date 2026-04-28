import torch 
from torch.utils.data import Dataset
import numpy as np 
from transformers import AutoFeatureExtractor
import torchaudio

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
        return input_values,torch.tensor(self.y[idx],dtype=torch.long),self.devices_id[idx]



class SSASTBreathDataset(Dataset):
    """Dataset for SSAST: produces raw log-mel spectrograms."""
    def __init__(self, X, y, devices_id, train=True,
                 sr=16000, n_mels=128, target_length=1024):
        self.X = X
        self.y = y
        self.devices_id = devices_id
        self.train = train
        self.sr = sr
        self.target_length = target_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=128,
            n_mels=n_mels, f_min=50, f_max=8000
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        wav = self.X[idx].copy()

        # Augmentation (same as baseline)
        if self.train:
            if np.random.random() < 0.5:
                wav = wav * np.random.uniform(0.9, 1.1)
                wav = wav + np.random.normal(0, 0.0001, wav.shape)

        # Waveform → log-mel spectrogram
        waveform = torch.FloatTensor(wav).unsqueeze(0)  # (1, samples)
        mel = self.mel_transform(waveform)               # (1, n_mels, time)
        mel = (mel + 1e-10).log()                        # log-mel
        mel = mel.squeeze(0).transpose(0, 1)             # (time, n_mels)

        # Pad or truncate to target_length
        if mel.shape[0] < self.target_length:
            pad = torch.zeros(self.target_length - mel.shape[0], mel.shape[1])
            mel = torch.cat([mel, pad], dim=0)
        else:
            mel = mel[:self.target_length, :]

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        return mel, torch.tensor(self.y[idx], dtype=torch.long), self.devices_id[idx]
