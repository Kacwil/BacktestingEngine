from models.ModelBase import Model_Base
import matplotlib.pylab as plt
import torch.nn as nn
import torch

#
#
# Most basic Feed-forward Neural Network
#
#

class FNNCore(nn.Module):
    
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.head = self.build_model()

    def forward(self, x):
        return self.head(x)

    def build_model(self):
        head = self.build_head()
        return head

    def build_head(self):
        head = nn.Sequential(
            nn.Linear(self.hp.n_features, 512),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(128, self.hp.n_targets)
        )
        return head
    
class FNN(Model_Base):
    def __init__(self, hp):
        super().__init__(hp)

    def _build_model(self):
        return FNNCore(self.hp)

    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
    
    def _loss_fn(self, predictions, targets):
        return nn.functional.mse_loss(predictions, targets)
    
    def debug_epoch(self, limit: int = 40):
        # --- headline numbers -------------------------------------------------
        tr_loss = self.log.train.avg_loss[-1]
        val_loss = self.log.val.avg_loss[-1]
        print(f"Training loss: {tr_loss:.4f}, Validation loss: {val_loss:.4f}")

        # --- fetch last validation batch --------------------------------------
        log = self.log.val
        preds   = log.predictions[-1].detach().cpu()        # shape [N, n_targets]
        targets = log.target_batches[-1].detach().cpu()     # same shape

        errors  = preds - targets
        mae     = errors.abs().mean(0)
        rmse    = (errors**2).mean(0).sqrt()

        n_targets = preds.size(1)

        # --- sample-by-sample debug print -------------------------------------
        hdr_pred  = " ".join([f"P{i}" for i in range(n_targets)])
        hdr_targ  = " ".join([f"T{i}" for i in range(n_targets)])
        hdr_err   = " ".join([f"E{i}" for i in range(n_targets)])
        print(f"\n{'Idx':<5} {hdr_pred:<25} {hdr_targ:<25} {hdr_err}")
        print("-" * 70)

        show = min(limit, len(preds))
        for idx in range(show):
            p = preds[idx].numpy()
            t = targets[idx].numpy()
            e = errors[idx].numpy()
            # keep numbers concise
            fmt = lambda arr: " ".join(f"{x:>7.3f}" for x in arr)
            print(f"{idx:<5} {fmt(p):<25} {fmt(t):<25} {fmt(e)}")

        # --- aggregate metrics -------------------------------------------------
        print("\nPer-target MAE :", " ".join(f"{m:.4f}" for m in mae))
        print("Per-target RMSE:", " ".join(f"{r:.4f}" for r in rmse))
    
    def debug_training(self):
        #Plot best validation epoch
        print(f"Best epoch: {self.log.val.best_epoch}, {self.log.val.avg_loss[self.log.val.best_epoch]}")

        return None


    def debug_test(self, max_samples: int = 1000):
        # 1. stack batches → [N, 5]
        preds   = torch.cat(self.log.test.predictions, dim=0)[:max_samples]
        targets = torch.cat(self.log.test.target_batches, dim=0)[:max_samples]

        n_out   = preds.shape[1]          # 5  (open, high, low, close, volume)
        labels  = ["Open", "High", "Low", "Close", "Volume"]

        fig, axes = plt.subplots(
            n_out, 1, figsize=(12, 2.6 * n_out), sharex=True, tight_layout=True
        )

        for i, ax in enumerate(axes):
            ax.plot(preds[:, i],   label="Pred",  alpha=0.7)
            ax.plot(targets[:, i], label="Target", alpha=0.7, linestyle="--", linewidth=1)
            ax.set_ylabel(labels[i])
            ax.grid(True)
            ax.legend()

        axes[0].set_title("Test set: Predictions vs Targets")
        axes[-1].set_xlabel("Sample Index")
        plt.show()