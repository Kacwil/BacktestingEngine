from database.db_manager import DB_Manager
from database.feature_extractor import extract_feature_targets
from datapipe.datapipe import datapipe
from datapipe.PipelineParameters import Pipeline_Params
from models.LSTM import LSTM
from models.HyperParameters import HyperParams


def main():
    dbm = DB_Manager()
    tfs = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"}

    table_names = [dbm.db.table_name("BTC/USDT", tf) for tf in tfs]

    raw_datas = {table_name:dbm.db.select_data(table_name) for table_name in table_names}

    datas = {table_name:extract_feature_targets(raw_data) for table_name, raw_data in raw_datas.items()}


    for r in datas:
        print(r)

    params = Pipeline_Params(feature_seq_len=10, feature_seq_stride=1, target_seq_len=1, target_seq_stride=1)
    datasets = {n:datapipe(data[0].drop(["timestamp", "close"], axis=1), data[1]["class"], params) for n,data in datas.items()}

    hp = HyperParams(name="V1", n_features=33, n_targets=3)
    for n, ds in datasets.items():
        print("------")
        print(n)
        lstm = LSTM(hp)
        lstm.train_model(ds)


if __name__ == "__main__":
    main()