import torch
import os
from torch.utils.data import DataLoader
from utils.dataclasses import TrainingLog, DataSplit
from models.HyperParameters import HyperParams
    
class Model_Base():
    '''
    Creates universal torch model components.
    Basic training interface.
    '''
    def __init__(self, hp: HyperParams):
        #ID
        self.name = hp.name
        self.path = f"torch_saves/{hp.name}.pth"
        self.hp = hp

        #Data
        self.training_log = TrainingLog()

        #Functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()

    def _build_model(self):
        raise NotImplementedError("Subclasses should implement _build_model()")

    def _build_optimizer(self):
        raise NotImplementedError("Subclasses should implement build_optimizer()")
    
    def _loss_fn(self):
        raise NotImplementedError("Subclasses should implement loss_fn()")
    
    def debug_epoch(self):
        raise NotImplementedError("Subclasses should implement debug_epoch()")
    
    def debug_training(self):
        raise NotImplementedError("Subclasses should implement debug_training()")

    def _create_dataloaders(self, ds:DataSplit) -> DataSplit:
        shuffle = self.hp.shuffle_training

        return DataSplit(*[
            DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=(name == "train" and shuffle))
            for name, dataset in ds.items()
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

    def train_model(self, dataset:DataSplit) -> None:
        """
        Trains the model using "train" and "val" dataloaders.
        """
        dataloaders = self._create_dataloaders(dataset)

        for epoch in range(self.hp.epochs):

            print(f"Epoch: {epoch}")

            self._run_epoch(dataloaders.train, True)
            self._run_epoch(dataloaders.val, False)

            if self.training_log.val.best_epoch == epoch:
                self.save_model()

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

        total_loss, total_grad_norm = 0, 0

        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                predictions = self.model.forward(features)
                loss = self._loss_fn(predictions, targets)
                total_loss += loss.item()

                if train:
                    loss.backward()
                    total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                    self.optimizer.step()
        
        log.feature_batches.append(features.detach().cpu())
        log.target_batches.append(targets.detach().cpu())
        log.predictions.append(predictions.detach().cpu())
        log.avg_loss.append(total_loss/len(dataloader))

        if log.avg_loss[-1] < log.lowest_loss: 
            log.best_epoch = len(log.avg_loss) - 1
            log.lowest_loss = log.avg_loss[-1]
        
        if train:
            avg_grad_norm = total_grad_norm / len(dataloader)
            log.grad_norms.append(avg_grad_norm)

        return None
    