from utils.dataclasses import Data, DataSplit
from utils.HyperParameters import HyperParams
from datapipe.CustomDataset import CustomDataset
import torch
import pandas as pd
import copy

def datapipe(features:pd.DataFrame, targets:pd.DataFrame, params:HyperParams):
    """
    :Inputs: Dataframes of features and targets
    :Outputs: data_df, data_df_split, data_tensor_split, data_seq_tensor_split, datasets
    """

    data_df = Data(features, targets)
    data_df_split = split_data(data_df, params)
    data_df_split_norm = normalize_datasplit(data_df_split, params)
    data_tensor_split = convert_split_to_tensors(data_df_split_norm, params)
    data_seq_tensor_split = build_sequences(data_tensor_split, params)

    if params.sequential:
        datasets = make_datasets(data_seq_tensor_split)
    else:
        datasets = make_datasets(data_tensor_split)

    return data_df, data_df_split, data_tensor_split, data_df_split_norm, data_seq_tensor_split, datasets

def split_data(data:Data[pd.DataFrame], params:HyperParams) -> DataSplit:
    """Splits a df-data object into df-datasplit object"""

    n = len(data.features)

    train_size = int(n * params.train_ratio)
    val_size = int(n * params.val_ratio)

    train = Data(data.features[:train_size], data.targets[:train_size])
    val = Data(data.features[train_size:train_size + val_size], data.targets[train_size:train_size + val_size])
    test = Data(data.features[train_size + val_size:], data.targets[train_size + val_size:])

    return DataSplit(train, val, test)

def normalize_datasplit(datasplit:DataSplit, params):

    ds = copy.deepcopy(datasplit)

    for name, data in ds.items():
        for name, df in data.items():
            print(name)
            if name == "targets" and params.disable_normalize_targets:
                continue
            try:
                for col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / df[col].std().clip(1e-3)
            except:
                df = (df - df.mean()) / df.std().clip(1e-3)

    return ds

def convert_split_to_tensors(ds:DataSplit, params:HyperParams) -> DataSplit:
    """ Converts DataSplit of dataframes into Datasplit of tensors """
    result = []
    for _, data in ds.items():
        features = torch.tensor(data.features.values, dtype=torch.float32)
        targets = torch.tensor(data.targets.values, dtype=torch.float32)
        result.append(Data(features, targets))

    return DataSplit(*result)

def build_sequences(datasplit:DataSplit, params:HyperParams) -> DataSplit:
    """ Convert the 2d tensors in datasplit into 3d sequentiall tensors. """

    ds = copy.deepcopy(datasplit)

    start = (params.feature_seq_len - 1) * params.feature_seq_stride

    f_seq_len = params.feature_seq_len
    f_seq_stride = params.feature_seq_stride

    t_seq_len = params.target_seq_len
    t_seq_stride = params.target_seq_stride

    def _create_f_seq(tensor, seq_len, seq_stride, start, end):
        return torch.stack([tensor[i - (seq_len - 1) * seq_stride : i+1 : seq_stride]
                            for i in range(start, end)])

    def _create_t_seq(tensor, seq_len, seq_stride, start, end):
        return torch.stack([tensor[i : i + seq_len * seq_stride : seq_stride]
                            for i in range(start, end)])

    start = max((f_seq_len - 1) * f_seq_stride, 0)
    target_needed = (t_seq_len - 1) * t_seq_stride

    for _, data in ds.items():
        end = len(data.features) - target_needed
        data.features = _create_f_seq(data.features, f_seq_len, f_seq_stride, start, end)
        data.targets = _create_t_seq(data.targets, t_seq_len, t_seq_stride, start, end)
    return ds

def make_datasets(ds:DataSplit):
    """Create CustomDatasets from Datasplit"""
    datasets = []
    for name, data in ds.items():
        datasets.append(CustomDataset(data))

    return DataSplit(*datasets)


if __name__ == "__main__":
    from database.db_manager import DB_Manager
    import time
    dbm = DB_Manager()

    s = time.perf_counter()
    table_name = dbm.db.table_name("BTC/USDT", "1s")
    raw_data = dbm.db.select_data(table_name, dbm.db_data.tables[table_name].last - 100000 * 1000)

    params = HyperParams()
    datasets = datapipe(raw_data, raw_data, params)

    print(time.perf_counter() - s)
    print(len(raw_data))
    print(datasets.train.features.shape)