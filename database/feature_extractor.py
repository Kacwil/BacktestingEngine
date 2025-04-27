import pandas as pd
import talib as ta
import numpy as np

def extract_feature_targets(raw_data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts features and targets from ohlcv data
    """

    features = _extract_features(raw_data.copy())
    targets = _extract_targets(raw_data.copy())

    common_ts = set(features["timestamp"]) & set(targets["timestamp"])
    features = features[features["timestamp"].isin(common_ts)]
    targets = targets[targets["timestamp"].isin(common_ts)]

    features = features.sort_values("timestamp").reset_index(drop=True)
    targets = targets.sort_values("timestamp").reset_index(drop=True)

    return features, targets

def _extract_features(df:pd.DataFrame) -> pd.DataFrame:
    windows = [10, 30, 90, 270]
    normalization_window = 5

    for w in windows:
        df[f"ema{w}"] = normalize(ta.EMA(df["close"], w), normalization_window)
        df[f"rsi{w}"] = normalize(ta.RSI(df["close"], w), normalization_window)
        df[f"atr{w}"] = normalize(ta.ATR(df["high"], df["low"],df["close"], w), normalization_window)
        df[f"mfi{w}"] = normalize(ta.MFI(df["high"], df["low"], df["close"], df["volume"], w), normalization_window)

        macd = ta.MACD(df["close"], int(1.5*w), int(3*w), w)
        df[f"macd0_{w}"] = normalize(macd[0], normalization_window)
        df[f"macd1_{w}"] = normalize(macd[1], normalization_window)
        df[f"macd2_{w}"] = normalize(macd[2], normalization_window)

    timestamp_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    seconds_in_day = timestamp_dt.dt.hour * 3600 + timestamp_dt.dt.minute * 60 + timestamp_dt.dt.second
    seconds_in_week = timestamp_dt.dt.weekday * 86400 + seconds_in_day
    df["time_of_day_sin"] = np.sin(2 * np.pi * seconds_in_day / 86400)
    df["time_of_day_cos"] = np.cos(2 * np.pi * seconds_in_day / 86400)
    df["time_of_week_sin"] = np.sin(2 * np.pi * seconds_in_week / (86400 * 7))
    df["time_of_week_cos"] = np.cos(2 * np.pi * seconds_in_week / (86400 * 7))

    df["close"] = normalize(df["close"], normalization_window)
    df["volume"] = normalize(df["volume"], normalization_window)

    df = df.dropna()
    df = df.drop(["open", "high", "low"], axis=1)
    return df

def _extract_targets(raw_data:pd.DataFrame) -> pd.DataFrame:
    df = raw_data.drop(["open", "high", "low", "volume"], axis=1)
    th = 0.0000000001
    bins = [-np.inf, -th, th, np.inf]
    try:
        df["class"] = pd.qcut(df["close"].pct_change().shift(-1), q=3, labels=False)
    except:
        df["class"] = pd.cut(df["close"].pct_change().shift(-1), bins=bins, labels=[0,1,2])

    df = df.dropna()
    return df

def normalize(series, window=5, epsilon=1e-8, min_std=1e-4):

    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    norm = (series - rolling_mean) / (rolling_std + epsilon)

    #Low variance zones
    norm_rolling_mean = norm.rolling(5).mean()
    norm = norm.where(rolling_std > min_std, norm_rolling_mean)

    return norm

if __name__ == "__main__":
    from database.db_manager import DB_Manager
    import time
    dbm = DB_Manager()
    s = time.perf_counter()

    table_name = dbm.db.table_name("BTC/USDT", "1s")
    raw_data = dbm.db.select_data(table_name, dbm.db_data.tables[table_name].last - 100000 * 1000)

    x, y = extract_feature_targets(raw_data)

    print(time.perf_counter() - s)
    print(x.columns)

