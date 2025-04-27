from dataclasses import dataclass
from utils.enums import TICKERS, TIMEFRAMES

@dataclass
class Pipeline_Params:
    init_n_samples: int = 7*24*4*15*60
    end_n_samples: int = 10000
    ticker: TICKERS = TICKERS.BTC_USDT
    timeframe: TIMEFRAMES = TIMEFRAMES.S
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    feature_seq_len: int = 15 # SOMETHING IS BROKEN IF ITS SET TOO LOW probably pipeline sequence function 
    feature_seq_stride: int = 1
    target_seq_len: int  = 1
    target_seq_stride: int = 1
    save: bool = False
    is_classification: bool = False