from models.LearnerBase import BaseLearner, HyperParams
from datapipes.DataPipeline import DataPipeline
from models.V1 import V1
from models.HyperParameters import HyperParams

import matplotlib.pylab as plt
import torch.nn as nn
import torch

class Learner(BaseLearner):
    def __init__(self):

        hp = HyperParams()
        pl = DataPipeline()

        datasets = pl.run_datapipeline()
        df_data = pl.df_data

        self.model = V1(hp)

        super().__init__("V1", datasets, df_data, hp)
    
    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
    
    def _loss_fn(self, predictions, targets):
        return nn.functional.huber_loss(predictions, targets)
    
    def debug_epoch(self):
        print(f"Training loss: {self.training_log.train.avg_loss[-1]}, Validation loss: {self.training_log.val.avg_loss[-1]}")

        #Print a table of predictions/targets
        log = self.training_log.val
        limit = 50

        # Flatten all batches into one long list of rows
        preds = torch.cat(log.predictions[-1], dim=0).squeeze().cpu().numpy()
        targets = torch.cat(log.target_batches[-1], dim=0).squeeze().cpu().numpy()


        print(f"{'Prediction':>12} | {'Target':>8}")
        print("-" * 25)

        for p, t in zip(preds[:limit], targets[:limit]):
            print(f"{p:12.4f} | {t:8.4f}")
        return None
    
    def debug_training(self):
        #Plot best validation epoch
        max_samples=1000
        best_epoch = self.training_log.val.best_epoch

        # Flatten and trim to N samples
        preds = torch.cat(self.training_log.val.predictions[best_epoch], dim=0).squeeze()[:max_samples]
        targets = torch.cat(self.training_log.val.target_batches[best_epoch], dim=0).squeeze()[:max_samples]

        plt.figure(figsize=(12, 4))
        plt.plot(preds, label="Prediction", color="C0", alpha=0.7)
        plt.plot(targets, label="Target", color="C1", alpha=0.7, linestyle='--', linewidth=1)
        plt.title(f"Validation set: Predictions vs Targets — Epoch {best_epoch}")
        plt.xlabel("Sample Index")
        plt.ylabel("Cumulative Return (×100k)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout() 
        plt.show()

    def get_price_predictions_testset(self):
       return self._run_epoch(self.dataloaders.test, False, True)
    
l = Learner()
l.train_model()