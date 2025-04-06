from dataclasses import dataclass

@dataclass
class HyperParams:
    batch_size: int = 512
    epochs: int = 20
    lr: float = 0.001
    dropout: float = 0.20
    weight_decay: float = 1e-4
    leaky: float = 0.01 
    LSTM_hidden: int = 128 
    LSTM_layers: int = 2