from torch.utils.data import Dataset
import torch
from utils.dataclasses import Data

class CustomDataset(Dataset):
    def __init__(self, data:Data):
        "Takes in a tensor-Data object and creates a torch based dataset. The input tensors might be 2d or 3d.(sequences)"

        self.features:torch.tensor = data.features
        self.targets:torch.tensor = data.targets

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

    def n_features(self):
        return self.features.shape[-1]
    
    def n_targets(self):
        return self.targets.shape[-1]