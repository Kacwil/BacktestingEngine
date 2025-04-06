from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(
        self,
        features,
        targets,
        feature_cols=None,
        target_cols=None,
        input_window=1,
        output_window=1,
    ):
        self.input_window = input_window
        self.output_window = output_window

        # Store full sets for RL/debugging
        self.full_features = torch.tensor(features.values, dtype=torch.float32)
        self.full_targets = torch.tensor(targets.values, dtype=torch.float32)

        # Forecast-relevant subset
        self.feature_cols = feature_cols or list(range(features.shape[1]))
        self.target_cols = target_cols or list(range(targets.shape[1]))

        self.features = self.full_features[:, self.feature_cols]
        self.targets = self.full_targets[:, self.target_cols]

        self.max_index = len(self.features) - self.input_window - self.output_window + 1
        if self.max_index <= 0:
            raise ValueError("Not enough data to generate any sequences with given input/output windows.")

        # Precompute sequences
        self.input_sequences = torch.stack([
            self.features[i : i + self.input_window] for i in range(self.max_index)
        ])
        self.target_sequences = torch.stack([
            self.targets[i + self.input_window : i + self.input_window + self.output_window]
            for i in range(self.max_index)
        ])

        if self.output_window == 1:
            self.target_sequences = self.target_sequences.squeeze(1)

        if self.input_window == 1:
            self.input_sequences = self.input_sequences.squeeze(1)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]
    
    def __len__(self):
        return max(0, self.max_index)

    def n_features(self):
        return self.features.shape[1]

    def n_targets(self):
        return self.targets.shape[1]