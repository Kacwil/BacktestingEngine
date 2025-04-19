from dataclasses import dataclass

@dataclass
class HyperParams:
    batch_size: int = 100
    epochs: int = 8
    lr: float = 0.001
    dropout: float = 0.30
    weight_decay: float = 1e-4
    leaky: float = 0.01 

    n_features: int = 10
    n_targets: int = 5
    LSTM: bool = True
    LSTM_hidden: int = 128 
    LSTM_layers: int = 2