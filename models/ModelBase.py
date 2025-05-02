import torch
import os
from torch.utils.data import DataLoader
from utils.dataclasses import TrainingLog, Log, DataSplit
from utils.HyperParameters import HyperParams
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

    
class Model_Base():
    '''
    Creates universal torch model components.
    Basic training interface.

    Subclasses must implement:
    _build_model(), _build_optimizer(), _loss_fn()

    Subclasses can implement:
    debug_epoch(), debug_training(), debug_test()
    '''
    def __init__(self, hp: HyperParams):
        #ID
        self.name = hp.name
        self.path = f"torch_saves/{hp.name}.pth"
        self.hp = hp

        #Data
        self.log = TrainingLog()
        self.test_log = Log()

        #Functions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scaler = torch.GradScaler(self.device)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.75, patience=5)

    def _build_model(self):
        raise NotImplementedError("Subclasses should implement _build_model()")

    def _build_optimizer(self):
        raise NotImplementedError("Subclasses should implement _build_optimizer()")
    
    def _loss_fn(self):
        raise NotImplementedError("Subclasses should implement _loss_fn()")
    
    def debug_epoch(self):
        print("Subclasses should implement debug_epoch()")
    
    def debug_training(self):
        print("Subclasses should implement debug_training()")

    def _create_dataloaders(self, ds:DataSplit) -> DataSplit:
        shuffle = self.hp.shuffle_training

        return DataSplit(*[
            DataLoader(dataset, batch_size=self.hp.batch_size, shuffle=(name == "train" and shuffle), drop_last=True)
            for name, dataset in ds.items()
        ])

    def save_model(self):
        try:
            os.makedirs("torch_saves", exist_ok=True)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, self.path)
        except:
            raise Exception("Could not save the model weights")

    def load_model(self):
        try:
            checkpoint = torch.load(self.path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.model.to(self.device)
        except:
            raise Exception("Could not load the model weights")

    def train_model(self, dataset:DataSplit) -> None:
        """
        Trains the model using "train" and "val" dataloaders.
        Saves the best epoch
        """
        dataloaders = self._create_dataloaders(dataset)

        for epoch in range(self.hp.epochs):

            print(f"Epoch: {epoch}")

            self._run_epoch(dataloaders.train, "train")
            self._run_epoch(dataloaders.val, "val")

            if self.log.val.best_epoch == epoch:
                self.save_model()

            self.debug_epoch()
        self.debug_training()

        return None

    def _run_epoch(self, dataloader:DataLoader, dataset_name:str):


        start_timer = time.perf_counter()

        train, val = False, False
        if dataset_name == "train":
            self.model.train()
            self.optimizer.zero_grad()
            log = self.log.train
            train = True
        elif dataset_name == "val":
            self.model.eval()
            log = self.log.val
            val = True
        elif dataset_name == "test":
            self.model.eval()
            log = self.log.test
        else:
            raise Exception("Invalid dataset_name passed to _run_epoch.")

        total_loss, total_grad_norm = 0, 0

        for features, targets in dataloader:

            features = features.to(self.device)
            targets = targets.to(self.device)
            
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                with torch.autocast(str(self.device)):
                    predictions = self.model.forward(features)
                    loss = self._loss_fn(predictions, targets)
                    total_loss += loss.item()

                    if train:
                        self.scaler.scale(loss).backward()

                        self.scaler.unscale_(self.optimizer)  # unscale before clipping
                        total_grad_norm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.5)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

        if val: 
            self.scheduler.step(total_loss/len(dataloader))
            print(f"LR: {self.scheduler.optimizer.param_groups[0]['lr']}")

        log.epoch_times.append(time.perf_counter() - start_timer)
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
    
    def test_model(self, dataset:DataSplit):
        """
        Runs the testset once.
        Loads saved weights
        """
        self.load_model()
        dataloaders = self._create_dataloaders(dataset)
        self._run_epoch(dataloaders.test, "test") 
        self.debug_test()

    def debug_test(self):
        print("debug_test() not implemented")
        return None


