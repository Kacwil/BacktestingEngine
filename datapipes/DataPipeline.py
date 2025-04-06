from database.db_client import DB_Client
from utils.enums import TICKERS, TIMEFRAMES
from utils.dataclasses import DF_Data, Split, Pipeline_Params
from datapipes.CustomDataset import CustomDataset
import pandas as pd

class DataPipeline():
    def __init__(self):
        """Fetches data from db and creates datasets for the neural network. Simply access the class variables"""
        self.db_client = DB_Client()
        self.params = Pipeline_Params()

        #placeholders
        ticker = TICKERS.BTC_USDT 
        timeframe = TIMEFRAMES.S

        self.df_data = self.get_data(ticker, timeframe)
        self.datasets = self.create_datasets(self.df_data)

    def get_data(self, ticker, timeframe) -> DF_Data:
        """Fetches and splits data."""
        data = self.db_client.select_features_targets(ticker, timeframe, limit=self.params.init_n_samples)

        features = pd.DataFrame(data[0]).drop(columns=0, axis=1).reset_index(drop=True)
        targets = pd.DataFrame(data[1]).drop(columns=0, axis=1).reset_index(drop=True)

        targets["scaled_cum_return"] = targets[2] * 10000000

        print(targets)

        data = DF_Data(self.split_data(features), self.split_data(targets))
        
        return data
    
    def split_data(self, df) -> Split[pd.DataFrame]:
        """Splits a single df into train/val/test."""

        train_size = int(len(df) * self.params.train_ratio)
        val_size = int(len(df) * self.params.val_ratio)

        train = df[:train_size]
        val = df[train_size:train_size + val_size]
        test = df[train_size + val_size:]

        return Split(train, val, test)
    
    def create_datasets(self, df_data:DF_Data) -> Split[CustomDataset]:
        """Creates DatasetSplit based on torch datasets"""

        return Split(*[CustomDataset(features, targets, [i for i in range(1, features.shape[1])], [-1], 10, 1)
                     for _, features, targets in df_data.items()])