import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from utils.dataclasses import Split, TrainingLog, DF_Data
from datapipes.CustomDataset import CustomDataset
from models.HyperParameters import HyperParams

    
class BaseLearner():
    '''
    A base class for neural networks to inherit from.
    '''
    def __init__(self, name, datasets: Split[CustomDataset], df_data:DF_Data , hp: HyperParams):
        #ID
        self.name = name
        self.path = f"torch_saves/{name}.pth"
        self.hp = hp

        #Data
        self.datasets = datasets
        self.df_data = df_data
        self.training_log = TrainingLog()

        #Functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.dataloaders = self._create_dataloaders()

    
    def _build_optimizer(self, **kwargs):
        raise NotImplementedError("Subclasses should implement build_optimizer()")
    
    def _loss_fn(self, predictions, targets, *args):
        raise NotImplementedError("Subclasses should implement loss_fn()")
    
    def debug_epoch(self, *args):
        raise NotImplementedError("Subclasses should implement debug_epoch()")
    
    def debug_training(self, *args):
        raise NotImplementedError("Subclasses should implement debug_training()")

    def _create_dataloaders(self):
        return Split(*[
            DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=(name == "train"))
            for name, dataset in self.datasets.items()
        ])


    def save_model(self):
        os.makedirs("torch_saves", exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, self.path)

    def load_model(self):
        checkpoint = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        self.optimizer.to(self.device)

    def train_model(self) -> None:
        """
        Trains the model using "train" and "val" dataloaders.
        """

        for epoch in range(self.hp.epochs):

            print(f"Epoch: {epoch}")

            self._run_epoch(self.dataloaders.train, True)
            self._run_epoch(self.dataloaders.val, False)

            if self.training_log.val.best_epoch == epoch:
                #self.save_model()
                pass

            self.debug_epoch()
        self.debug_training()

        return None


    def _run_epoch(self, dataloader:DataLoader, train:bool):

        if train:
            self.model.train()
            log = self.training_log.train
        else:
            self.model.eval()
            log = self.training_log.val

        total_loss = 0
        total_grad_norm = 0
        all_features = []
        all_targets = []
        all_predictions = []

        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                predictions = self.model.forward(features)
                loss = self._loss_fn(predictions, targets)
                total_loss += loss.item()

                all_features.append(features.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_predictions.append(predictions.detach().cpu())

                if train:
                    loss.backward()
                    total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optimizer.step()
        
        log.feature_batches.append(all_features)
        log.target_batches.append(all_targets)
        log.predictions.append(all_predictions)
        log.avg_loss.append(total_loss/len(dataloader))

        if log.avg_loss[-1] < log.lowest_loss: 
            log.best_epoch = len(log.avg_loss) - 1
            log.lowest_loss = log.avg_loss[-1]
        
        if train:
            avg_grad_norm = total_grad_norm / len(dataloader)
            log.grad_norms.append(avg_grad_norm)

        return None
    