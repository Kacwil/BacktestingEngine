from models.ModelBase import Model_Base
from models.LSTMCore import LSTMCore
import matplotlib.pylab as plt
import torch.nn as nn
import torch

class LSTM(Model_Base):
    def __init__(self, hp):
        super().__init__(hp)

    def _build_model(self):
        return LSTMCore(self.hp)

    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
    
    def _loss_fn(self, predictions, targets):
        targets = targets.long().squeeze()
        return nn.functional.cross_entropy(predictions, targets)
    
    def debug_epoch(self):
        print(f"Training loss: {self.training_log.train.avg_loss[-1]}, Validation loss: {self.training_log.val.avg_loss[-1]}")

        log = self.training_log.val
        limit = 40

        predictions = torch.argmax(log.predictions[-1], axis=1)
        targets = log.target_batches[-1].squeeze()

        print(f"\n{'Idx':<5} {'Pred':<5} {'Target':<6} {'Wrong?'}")
        print("-" * 30)
        for idx in range(min(limit, len(predictions))):
            pred = predictions[idx].item()
            target = targets[idx].item()
            wrong = "X" if pred != target else ""
            print(f"{idx:<5} {pred:<5} {target:<6} {wrong}")

    
    def debug_training(self):
        #Plot best validation epoch

        return None
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
    