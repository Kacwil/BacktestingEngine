from dataclasses import dataclass
from utils.enums import TICKERS, TIMEFRAMES

@dataclass
class HyperParams:
    # --- Pipeline Parameters ---
    init_n_samples: int = 7*24*4*15*60
    end_n_samples: int = 10000
    ticker: TICKERS = TICKERS.BTC_USDT
    timeframe: TIMEFRAMES = TIMEFRAMES.S
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    sequential: bool = False
    feature_seq_len: int = 15
    feature_seq_stride: int = 1
    target_seq_len: int  = 1
    target_seq_stride: int = 1
    save: bool = False
    is_classification: bool = False
    disable_normalize_targets: bool = True

    # --- Model Parameters ---
    name:str = "default"
    batch_size: int = 5096
    epochs: int = 8
    lr: float = 0.001
    dropout: float = 0.25
    weight_decay: float = 1e-4
    leaky: float = 0.01 
    shuffle_training: bool = True

    n_features: int = 10
    n_targets: int = 5
    LSTM: bool = True
    LSTM_hidden: int = 128
    LSTM_layers: int = 2