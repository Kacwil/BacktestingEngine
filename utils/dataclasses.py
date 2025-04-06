from dataclasses import dataclass, field, fields
import pandas as pd
from utils.enums import TICKERS, TIMEFRAMES
from typing import Generic, TypeVar, Iterator, Tuple

T = TypeVar("T")

@dataclass
class Split(Generic[T]):
    train: T
    val: T
    test: T

    def items(self) -> Iterator[Tuple[str, T]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

@dataclass
class DF_Data:
    features: Split[pd.DataFrame]
    targets: Split[pd.DataFrame]

    def items(self) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
        for name in ["train", "val", "test"]:
            yield name, getattr(self.features, name), getattr(self.targets, name)

@dataclass
class Log:
    feature_batches: list = field(default_factory=list)
    target_batches: list = field(default_factory=list)
    predictions: list = field(default_factory=list)

    avg_loss: list = field(default_factory=list)
    best_epoch: int = 0
    lowest_loss: float = 1e10

    grad_norms: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    epoch_times: list = field(default_factory=list)
    events: list = field(default_factory=list)

@dataclass
class TrainingLog:
    train: Log = field(default_factory=Log)
    val: Log = field(default_factory=Log)


@dataclass
class Pipeline_Params:
    init_n_samples: int = 7*24*4*15*60
    end_n_samples: int = 50000
    ticker: TICKERS = TICKERS.BTC_USDT
    timeframe: TIMEFRAMES = TIMEFRAMES.S
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    sequential: bool = True
    stride: int = 2
    seq_length: int = 5
    save: bool = False
    is_classification: bool = False
