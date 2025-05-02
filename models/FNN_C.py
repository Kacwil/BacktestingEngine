from models.ModelBase import Model_Base
import matplotlib.pylab as plt
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix
#
#
# Feed-forward Neural Network Classifier
#
#

class FNNCCore(nn.Module):
    """
    Feed-forward network for a single binary target.
    """
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.head = nn.Sequential(
            nn.Linear(self.hp.n_features, 256),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),

            nn.Linear(256, 256),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),

            nn.Linear(256, 128),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(x)


class FNN_C(Model_Base):

    def __init__(self, hp):
        super().__init__(hp)

    def _build_model(self):
        return FNNCCore(self.hp)

    def _build_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay
        )

    def _loss_fn(self, predictions, targets):
        return nn.functional.binary_cross_entropy_with_logits(predictions, targets.unsqueeze(1))
    


    def debug_epoch(self):
        # --- headline numbers -------------------------------------------------
        tr_loss = self.log.train.avg_loss[-1]
        val_loss = self.log.val.avg_loss[-1]
        print(f"Training loss: {tr_loss:.4f}, Validation loss: {val_loss:.4f}")

        p = self.log.val.predictions[-1]
        t = self.log.val.target_batches[-1]

        # binary predictions
        pred_labels = (torch.sigmoid(p.squeeze()) > 0.5).long().cpu().numpy()
        true_labels = t.squeeze().long().cpu().numpy()

        # confusion matrix: rows = true class, cols = predicted class
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

        # optional: overall accuracy for reference
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        print(f"Accuracy: {acc:.4f}")
        

    
    def debug_training(self):
        #Plot best validation epoch
        print(f"Best epoch: {self.log.val.best_epoch}, {self.log.val.avg_loss[self.log.val.best_epoch]}")


        return None


    def debug_test(self, max_samples: int = 1000):
        # 1. stack batches → [N, 5]
        preds   = torch.cat(self.log.test.predictions, dim=0)[:max_samples]
        targets = torch.cat(self.log.test.target_batches, dim=0)[:max_samples]