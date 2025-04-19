import pandas as pd
import talib as ta
import numpy as np
import datetime

class Feature_Extractor():
    """Makes some basic normalized features and targets from data"""
    def __init__(self):
        pass

    def create_features_targets(self,data):
        df_features = self._create_features(data.copy())
        df_targets = self._create_targets(data.copy())

        #Crop
        common_ts = set(df_features["timestamp"]) & set(df_targets["timestamp"])
        df_features = df_features[df_features["timestamp"].isin(common_ts)]
        df_targets = df_targets[df_targets["timestamp"].isin(common_ts)]

        df_features = df_features.sort_values("timestamp").reset_index(drop=True)
        df_targets = df_targets.sort_values("timestamp").reset_index(drop=True)

        return df_features, df_targets

    def _create_features(self, df):

        windows = [10, 30, 90, 270, 810, 2430]
        normalization_window = 30

        for w in windows:
            df[f"ema{w}"] = self.normalize(ta.EMA(df["close"], w), normalization_window)
            df[f"rsi{w}"] = self.normalize(ta.RSI(df["close"], w), normalization_window)
            df[f"atr{w}"] = self.normalize(ta.ATR(df["high"], df["low"],df["close"], w), normalization_window)
            df[f"mfi{w}"] = self.normalize(ta.MFI(df["high"], df["low"], df["close"], df["volume"], w), normalization_window)

            if w == 2430: break

            macd = ta.MACD(df["close"], int(1.5*w), int(3*w), w)
            df[f"macd0_{w}"] = self.normalize(macd[0], normalization_window)
            df[f"macd1_{w}"] = self.normalize(macd[1], normalization_window)
            df[f"macd2_{w}"] = self.normalize(macd[2], normalization_window)

        timestamp_dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        seconds_in_day = timestamp_dt.dt.hour * 3600 + timestamp_dt.dt.minute * 60 + timestamp_dt.dt.second
        seconds_in_week = timestamp_dt.dt.weekday * 86400 + seconds_in_day
        df["time_of_day_sin"] = np.sin(2 * np.pi * seconds_in_day / 86400)
        df["time_of_day_cos"] = np.cos(2 * np.pi * seconds_in_day / 86400)
        df["time_of_week_sin"] = np.sin(2 * np.pi * seconds_in_week / (86400 * 7))
        df["time_of_week_cos"] = np.cos(2 * np.pi * seconds_in_week / (86400 * 7))

        df = df.dropna()
        df = df.drop(["open", "high", "low"], axis=1)
        return df

    def _create_targets(self, df):
        df = df.drop(["open", "high", "low", "volume"], axis=1)
        
        df["rets"] = df["close"].pct_change()

        #Trick to vectorize without using np.prod
        df["log_rets"] = np.log(df["rets"] + 1)

        df["cum_rets"] = (np.exp(df["log_rets"].rolling(window=60).sum().shift(-59)) - 1)
        df["n_cum_rets"] = self.normalize(df["cum_rets"], 15)
        df["ewm_cum_rets"] = self.normalize_ewm(df["cum_rets"], 15)
        df["minmax_cum_rets"] = self.normalize_minmax(df["cum_rets"], 15)

        df = df.dropna()
        return df

    def normalize(self, series, window=30, epsilon=1e-8, min_std=1e-4):
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()

        norm = (series - rolling_mean) / (rolling_std + epsilon)

        # Fallback for low variance zones
        norm_rolling_mean = norm.rolling(5).mean()
        norm = norm.where(rolling_std > min_std, norm_rolling_mean)

        return norm

    def normalize_ewm(self, series, span=30, epsilon=1e-8, min_std=1e-4):
        rolling_mean = series.ewm(span=span).mean()
        rolling_std = series.ewm(span=span).std()

        norm = (series - rolling_mean) / (rolling_std + epsilon)

        # Fallback for low variance zones
        fallback = norm.ewm(span=5).mean()
        norm = norm.where(rolling_std > min_std, fallback)

        return norm
    
    def normalize_minmax(self, series, window=30, epsilon=1e-8):
        rolling_min = series.rolling(window).min()
        rolling_max = series.rolling(window).max()       
        norm = (series - rolling_min) / (rolling_max - rolling_min + epsilon)
        return (norm - 0.5) * 2




#dbm = DB_Manager()
#data = dbm.db.select_ohlcv_data("BTC/USDT", "1s")

#fe = Feature_Extractor()
#df = fe.create_targets(data)
#df = fe.create_features(df)
