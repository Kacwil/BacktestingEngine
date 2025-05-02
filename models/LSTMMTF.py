from models.ModelBase import Model_Base
import matplotlib.pylab as plt
import torch.nn as nn
import torch
from sklearn.metrics import confusion_matrix

class LSTMMTFCore(nn.Module):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder1, self.encoder2, self.encoder3, self.encoder, self.head = self.build_model()

    def forward(self, x):
        encoded1, _ = self.encoder1(x[:,:,:10])
        encoded2, _ = self.encoder2(x[:,:,10:20])
        encoded3, _ = self.encoder3(x[:,:,20:])

        encoded = torch.cat((encoded1, encoded2, encoded3), dim=-1)
        encoded, _ = self.encoder(encoded)
        return self.head(encoded[:, -1, :])

    def build_model(self):
        encoder1 = nn.LSTM(input_size=self.hp.n_features, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
        encoder2 = nn.LSTM(input_size=self.hp.n_features, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
        encoder3 = nn.LSTM(input_size=self.hp.n_features, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
        encoder = nn.LSTM(input_size=self.hp.LSTM_hidden * 3, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
        head = self.build_head()
        return encoder1, encoder2, encoder3, encoder, head

    def build_head(self):
        head = nn.Sequential(
            nn.Linear(self.hp.LSTM_hidden, 128),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(64, self.hp.n_targets)
        )
        return head

class LSTMMTF(Model_Base):
    def __init__(self, hp):
        super().__init__(hp)

    def _build_model(self):
        return LSTMMTFCore(self.hp)

    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hp.lr, weight_decay=self.hp.weight_decay)
    
    def _loss_fn(self, predictions, targets):
        targets = targets.squeeze()
        class_weights = torch.tensor([0.95, 1.0, 1.0, 1.0, 0.95]).to(self.device)
        targets = targets.long()
        return nn.functional.cross_entropy(predictions, targets, weight=class_weights)
    
    def debug_epoch(self):

        print(f"Training loss: {self.log.train.avg_loss[-1]}, Validation loss: {self.log.val.avg_loss[-1]}")

        log = self.log.val
        limit = 40

        predictions = torch.argmax(log.predictions[-1], axis=1)
        targets = log.target_batches[-1].squeeze()

        cm = confusion_matrix(targets.cpu(), predictions.cpu())
        accuracy = (predictions == targets).float().mean().item()

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)

        print("\nAccuracy:")
        print(accuracy)

        return None
        print(f"\n{'Idx':<5} {'Pred':<5} {'Target':<6} {'Wrong?'}")
        print("-" * 30)
        for idx in range(min(limit, len(predictions))):
            pred = predictions[idx].item()
            target = targets[idx].item()
            wrong = "X" if pred != target else ""
            print(f"{idx:<5} {pred:<5} {target:<6} {wrong}")

    
    def debug_training(self):
        #Plot best validation epoch

        max_samples=100
        best_epoch = self.log.val.best_epoch

        # Flatten and trim to N samples
        preds = torch.cat(self.log.val.predictions[best_epoch], dim=0).squeeze()[:max_samples]
        targets = torch.cat(self.log.val.target_batches[best_epoch], dim=0).squeeze()[:max_samples]

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
    
    def test_model(self, ds):
        self.load_model()

        dataloaders = self._create_dataloaders(ds)
        self._run_epoch(dataloaders.test, "test")

        print("Running Testset")

        predictions = torch.argmax(torch.cat(self.log.test.predictions, dim=-1), axis=1)
        targets = torch.squeeze(torch.cat(self.log.test.target_batches, dim=-1))

        cm = confusion_matrix(targets.cpu(), predictions.cpu())
        accuracy = (predictions == targets).float().mean().item()

        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)

        print("\nAccuracy:")
        print(accuracy)