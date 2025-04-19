import torch.nn as nn

class V1(nn.Module):

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.encoder, self.head = self.build_model()

    def forward(self, x):
        if self.hp.LSTM:
            encoded, _ = self.encoder(x)
            last = encoded[:, -1]
            return self.head(last)
        else:
            return self.head(x)

    def build_model(self):
        if self.hp.LSTM:
            encoder = nn.LSTM(input_size=self.hp.n_features, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
        else:
            encoder = None
            self.hp.LSTM_hidden = self.hp.n_features

        head = self.build_head()
        return encoder, head

    def build_head(self):
        head = nn.Sequential(
            nn.Linear(self.hp.LSTM_hidden, 128),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(self.hp.leaky),
            nn.Dropout(self.hp.dropout),
            nn.Linear(32, self.hp.n_targets)
        )
        return head