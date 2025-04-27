from database.db_manager import DB_Manager
from utils.utils import timeframe_to_timeskip
import pandas as pd
import asyncio
import inspect
import talib
import copy
import matplotlib as plt


class Feature_Generator():
    def __init__(self):
        self.dbm = DB_Manager()
        self.ticker = "BTC/USDT"
        self.tf = "1s"

    def batch_generator(self, ticker, tf, start=None):
        table_name = self.dbm.db.table_name(ticker, tf)
        start = start or self.dbm.db_data.tables[table_name].first
        time_skip = timeframe_to_timeskip(tf)

        batch_size = 60 * 60 * 24 // 2  # Half a day in seconds
        batch_time_skip = (batch_size - 1) * time_skip

        df = self.dbm.db.select_data(table_name, start=start, end=start+batch_time_skip)
        start = start + batch_time_skip + time_skip
        last = None

        while True:

            new_df = self.dbm.db.select_data(table_name, start=start, end=start+batch_time_skip)
            start = start + batch_time_skip + time_skip
            prev_df = df.tail((batch_size * 2) - len(new_df))
            last = prev_df.tail(1)["timestamp"].values[0]

            if last == self.dbm.db_data.tables[table_name].last:
                break

            df = pd.concat([prev_df, new_df], ignore_index=True)
            yield df

    


def generate_data(batch_df:pd.DataFrame):

    dfs = [resample_candles(batch_df, window) for window in [1, 5, 10, 30, 60, 120, 300, 900]]
    targets = create_targets(batch_df)

    ta_funcs = get_talib_functions()
    args_grouped = sort_by_args(ta_funcs)
    features = {}

    for df in dfs:
        df_window = df.columns[0].split("_")[0]
        features[df_window] = {}
        for group, func_list in args_grouped.items():

            args = get_input_args(group, df)
            for (func_str, func) in func_list:
                result = func(**args)
                if len(result) == len(df):
                    features[df_window][func_str] = result
                else:
                    i = 0
                    for res in result:
                        features[df_window][func_str + str(i)] = res
                        i += 1

        features[df_window] = pd.DataFrame(features[df_window])

    for window, features in features.items():
        print(features.reset_index(inplace=True))
        print(targets["ret_60"])
        co = features.corrwith(targets["ret_10"])
        print("--------")
        print(window)
        print(co.abs().nlargest(20))

    return None

    print(batch_df)


    targets = create_targets(batch_df)
    features_c, features_b = create_features(batch_df)

    co = features_c.corr()
    tco = features_c.corrwith(targets["ret_10"]).abs()

    cob = features_b.corr()
    tcob = features_b.corrwith(targets["ret_10"]).abs()

    return tco, tcob

def sort_by_args(ta_funcs) -> dict[str, tuple[str, object]]:
    args_groups = {}

    for func_name, (func, sig) in ta_funcs.items():
        args = []
        for s, p in sig.parameters.items():
            if p.default is inspect.Parameter.empty or s in ["fastperiod", "slowperiod", "period"]:
                args.append(s)

        args = tuple(args)
        if args not in args_groups:
            args_groups[args] = []
        args_groups[args].append((func_name, func))

    return args_groups

def resample_candles(df, window_seconds):
    rule = f"{window_seconds}s"
    df = copy.deepcopy(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    

    resampled = pd.DataFrame()
    resampled[f"{window_seconds}_open"] = df["open"].resample(rule).first()
    resampled[f"{window_seconds}_high"] = df["high"].resample(rule).max()
    resampled[f"{window_seconds}_low"] = df["low"].resample(rule).min()
    resampled[f"{window_seconds}_close"] = df["close"].resample(rule).last()
    resampled[f"{window_seconds}_volume"] = df["volume"].resample(rule).sum()

    return resampled

def create_targets(batch_df):
    targets = pd.DataFrame()

    targets["ret_10"] = batch_df["close"].shift(-10)/batch_df["close"]
    targets["ret_20"] = batch_df["close"].shift(-20)/batch_df["close"]
    targets["ret_30"] = batch_df["close"].shift(-30)/batch_df["close"]
    targets["ret_60"] = batch_df["close"].shift(-60)/batch_df["close"]

    targets["c10"] = pd.qcut(targets["ret_10"], q=2, labels=False)
    targets["c20"] = pd.qcut(targets["ret_20"], q=3, labels=False)
    targets["c30"] = pd.qcut(targets["ret_30"], q=3, labels=False)
    targets["c60"] = pd.qcut(targets["ret_60"], q=5, labels=False)

    return targets

def create_features(df):
    features_cont_dict = {}
    features_bool_dict = {}

    timeperiods = [15, 30, 60, 150]
    fast_periods = [6, 12, 20, 100]
    slow_periods = [15, 26, 45, 250]

    for func_name, (func, sig) in get_talib_functions().items():

        try:
            args, timeperiod_flag = get_input_args(func, batch_df)

            for i in len(timeperiods):

                result = run_talib_function(func, args)

                print(result)

                if not timeperiod_flag:
                    break



        except Exception as e:
            print(f"{func_name}: {e}")

    features_cont = pd.DataFrame(features_cont_dict)
    features_cont["close"] = batch_df["close"]
    features_cont["volume"] = batch_df["volume"]
    features_cont.reset_index(inplace=True, drop=True)
    features_bool = pd.DataFrame(features_bool_dict)


    constant_cols = features_bool.columns[features_bool.nunique() <= 1]
    print("Constant columns to be dropped:", list(constant_cols))

    sparse_cols = features_bool.columns[features_bool.abs().mean() > 0.95]
    print("Spare columns to be dropped: ", list(sparse_cols))

    sparse_cols = features_bool.columns[features_bool.abs().mean() > 0.05]
    print("Spare columns to be dropped: ", list(sparse_cols))

    return features_cont, features_bool

def run_talib_function(func, args):
    return func(**args)

def get_talib_functions() -> dict[str, tuple[object, inspect.Signature]]:
    res = {}
    for group in talib.get_function_groups():

        if group in {"Math Operators", "Math Transform", "Statistic Functions"}:
            continue

        for func_name in talib.__function_groups__[group]:

            if func_name in {"MAVP"}:
                continue

            func = getattr(talib, func_name)
            sig = inspect.signature(func)

            res[func_name] = (func, sig)
    return res

def get_input_args(args_str:tuple[str], df:pd.DataFrame) -> dict[str, any]:
    args = {}
    for arg_str in args_str:
        df_window = df.columns[0].split("_")[0]
        try:
            args[arg_str] = df[df_window + "_" + arg_str]
        except:
            if arg_str == "real":
                args[arg_str] = df[df_window + "_close"]

    return args

def normalize(df: pd.DataFrame, window=30, eps=1e-6):
    """Rescale using past rolling window"""

    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()

    # Prevent division by near-zero std
    near_zero_std = rolling_std < eps
    normalized = (df - rolling_mean) / rolling_std
    normalized[near_zero_std] = rolling_mean[near_zero_std]

    return normalized

constant = []
def boolify(df:pd.DataFrame, func_name):
    """Expose constant features and rescale."""

    if df.mean() == 0 or abs(df.mean()) == 1 or abs(df.mean()) == 100:
        constant.append(func_name)

    if abs(df.max()) > 1:
        return df/100
    
    #print(constant)
    return df




fg = Feature_Generator()

t = pd.DataFrame()
total_tcob = pd.DataFrame()


for batch_df in fg.batch_generator("BTC/USDT", "1s"):
    t = pd.concat([t, batch_df]).reset_index(drop=True)
    #generate_data(batch_df)



