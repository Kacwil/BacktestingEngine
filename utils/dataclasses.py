from dataclasses import dataclass, field, fields
import pandas as pd
from utils.enums import TICKERS, TIMEFRAMES
from typing import Generic, TypeVar, Iterator, Tuple

T = TypeVar("T")

@dataclass
class Data(Generic[T]):
    """Stores some generic data, split into inputs and outputs for a NN"""
    features: T
    targets: T

    def items(self) -> Iterator[Tuple[str, T]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

@dataclass
class DataSplit:
    """Stores generic data objects in a train/val/test split"""
    train: Data
    val: Data
    test: Data

    def items(self) -> Iterator[Tuple[str, Data]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

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

