from database.db_client import DB_Client
from utils.dataclasses import Data, DataSplit
from datapipes.PipelineParameters import Pipeline_Params
from datapipes.CustomDataset import CustomDataset
import torch

class DataPipeline():
    def __init__(self):
        """Fetches data from db and creates datasets for the neural network. Simply access the class variables"""
        self.db_client = DB_Client()
        self.params = Pipeline_Params()

    def run_datapipeline(self):
        """Runs the whole datapipeline.
        Returns -> DataSplit(CustomDataset)
        """
        data = self._get_data(self.params.ticker, self.params.timeframe)
        df_split = self._split_data(data)
        tensor_split = self._create_tensors(df_split)
        seq_split = self._create_sequences(tensor_split)
        dataset_split = self._create_datasets(seq_split)
        return dataset_split

    def _get_data(self, ticker = "BTC/USDT", timeframe="1s"):
        """Fetch data from db."""
        return self.db_client.select_features_targets(ticker, timeframe, limit=self.params.init_n_samples)
    
    def _split_data(self, data) -> DataSplit:
        """Splits a dataframe-data object into dataframe-datasplit object"""
        n = len(data.features)

        train_size = int(n * self.params.train_ratio)
        val_size = int(n * self.params.val_ratio)

        train = Data(data.features[:train_size], data.targets[:train_size])
        val = Data(data.features[train_size:train_size + val_size], data.targets[train_size:train_size + val_size])
        test = Data(data.features[train_size + val_size:], data.targets[train_size + val_size:])

        return DataSplit(train, val, test)
    
    def _create_tensors(self, datasplit):
        result = []
        for _, data in datasplit.items():
            features = torch.tensor(data.features.values, dtype=torch.float32)
            targets = torch.tensor(data.targets.values, dtype=torch.float32)
            result.append(Data(features, targets))

        return DataSplit(*result)

    def _create_sequences(self, datasplit):

        start = (self.params.feature_seq_len - 1) * self.params.feature_seq_stride

        f_seq_len = self.params.feature_seq_len
        f_seq_stride = self.params.feature_seq_stride

        t_seq_len = self.params.target_seq_len
        t_seq_stride = self.params.target_seq_stride

        for _, data in datasplit.items():
            end = len(data.features) - (t_seq_len - 1) * t_seq_stride

            if f_seq_len > 1:
                data.features = self._create_f_seq(data.features, f_seq_len, f_seq_stride, start, end)
            else:
                data.features = data.features.squeeze(1)
            
            if t_seq_len > 1:
                data.targets = self._create_t_seq(data.targets, t_seq_len, t_seq_stride, start, end)
            else:
                data.targets = data.targets.squeeze(1)

        return datasplit
    
    def _create_f_seq(self, tensor, seq_len, seq_stride, start, end):
        return torch.stack([tensor[i - (seq_len - 1) * seq_stride : i+1 : seq_stride]
                            for i in range(start, end)])
    
    def _create_t_seq(self, tensor, seq_len, seq_stride, start, end):
        return torch.stack([tensor[i : i + seq_len * seq_stride : seq_stride]
                            for i in range(start, end)])
    
    def _create_datasets(self, datasplit):
        """Create CustomDatasets from Datasplit"""
        datasets = []
        for name, data in datasplit.items():
            datasets.append(CustomDataset(data))

        return DataSplit(*datasets)
    