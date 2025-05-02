from dataclasses import dataclass, field, fields
from typing import Generic, TypeVar, Iterator, Tuple

T = TypeVar("T")

@dataclass
class Data(Generic[T]):
    """Stores some generic data, split into features/targets for a NN"""
    features: T
    targets: T

    def items(self) -> Iterator[Tuple[str, T]]:
        for field in fields(self):
            yield field.name, getattr(self, field.name)

@dataclass
class DataSplit:
    """
    Stores generic Data objects in a train/val/test split.

    Data objects store features and targets in dataframes/tensors/datasets.
    """

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
    test: Log = field(default_factory=Log)


@dataclass
class TableData:
    name:str
    first:int = 1e100
    last:int = 0
    total_rows:int = 0
    expected_rows:int = 0

@dataclass
class DatabaseData:
    tables: dict[str, TableData] = field(default_factory=dict)

    def items(self) -> Iterator[Tuple[str, int, int, int, int]]:
        for name, data in self.tables:
            yield name, data.first, data.last, data.total_rows, data.expected_rows

