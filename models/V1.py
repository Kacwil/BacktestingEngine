import torch.nn as nn

class V1(nn.Module):

    def __init__(self, hp, input_size, output_size):
        super().__init__()
        self.hp = hp
        self.input_size = input_size
        self.output_size = output_size
        self.encoder, self.head = self.build_model()

    def forward(self, x):
        encoded, _ = self.encoder(x)
        last = encoded[:, -1]
        return self.head(last)

    def build_model(self):
        encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hp.LSTM_hidden, num_layers=self.hp.LSTM_layers, batch_first=True)
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
            nn.Linear(32, self.output_size)
        )
        return head