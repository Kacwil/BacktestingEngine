from database.db_manager import DB_Manager
from database.feature_extractor import extract_feature_targets
from datapipe.datapipe import datapipe
from datapipe.PipelineParameters import Pipeline_Params
from models.LSTM import LSTM
from models.HyperParameters import HyperParams


def main():
    dbm = DB_Manager()
    table_name = dbm.db.table_name("BTC/USDT", "1s")
    raw_data = dbm.db.select_data(table_name, dbm.db_data.tables[table_name].last - 100000 * 1000)

    features, targets = extract_feature_targets(raw_data)

    params = Pipeline_Params(feature_seq_len=5, feature_seq_stride=1, target_seq_len=1, target_seq_stride=1)
    datasets = datapipe(features.drop(["timestamp", "close"], axis=1), targets["class"], params)

    print(datasets.train.features.shape)
    print(datasets.train.targets.shape)

    hp = HyperParams(name="V1", n_features=40, n_targets=3)
    lstm = LSTM(hp)

    lstm.train_model(datasets)

if __name__ == "__main__":
    main()