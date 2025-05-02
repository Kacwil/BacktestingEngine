from database.db_manager import DB_Manager
from database.feature_extractor import extract_feature_targets
from datapipe.datapipe import datapipe
from utils.HyperParameters import HyperParams
from models.LSTM import LSTM


def main():
    dbm = DB_Manager()
    tfs = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "4h", "12h", "1d"}

    table_names = [dbm.db.table_name("BTC/USDT", tf) for tf in tfs]

    raw_datas = {table_name:dbm.db.select_data(table_name) for table_name in table_names}

    datas = {table_name:extract_feature_targets(raw_data) for table_name, raw_data in raw_datas.items()}

    params = HyperParams(feature_seq_len=10, feature_seq_stride=1, target_seq_len=1, target_seq_stride=1)
    datasets = {n:datapipe(data[0].drop(["timestamp", "close"], axis=1), data[1]["class"], params) for n,data in datas.items()}

    models = {}
    for n, ds in datasets.items():
        hp = HyperParams(name=n, n_features=33, n_targets=3, epochs=50)
        lstm = LSTM(hp)
        lstm.train_model(ds)
        models[n] = lstm

    for n, model in models.items():
        print(n, model.training_log.val.avg_loss, model.training_log.val.best_epoch)


if __name__ == "__main__":
    main()