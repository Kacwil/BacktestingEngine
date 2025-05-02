from database.db_manager import DB_Manager
from datapipe.datapipe import datapipe
from datapipe.CustomDataset import CustomDataset
from utils.dataclasses import Data, DataSplit
from models.LSTM import LSTM
from utils.HyperParameters import HyperParams
from models.ModelBase import Model_Base
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from models.LSTMMTF import LSTMMTF
import talib

def test():
    raw_data_list = init()
    datasets_list = []

    targets = pd.DataFrame(pd.to_datetime(raw_data_list[0]["timestamp"], unit="ms"))
    targets["class"] = pd.qcut(raw_data_list[0]["close"].pct_change().shift(-1), q=5, labels=False)
    targets.set_index("timestamp", inplace=True)

    for raw_data in raw_data_list:
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"], unit="ms")
        raw_data.set_index("timestamp", inplace=True)
        raw_data["ema"] = talib.EMA(raw_data["close"], 30)
        raw_data["obv"] = talib.OBV(raw_data["close"], raw_data["volume"])
        m1,m2,m3 = talib.MACD(raw_data["close"])
        raw_data["m1"] = m1
        raw_data["m2"] = m2
        raw_data["m3"] = m3
        raw_data.dropna(inplace=True)

    raw_data_list[1] = raw_data_list[1].resample("5min").ffill()  # Forward fill missing values
    raw_data_list[2] = raw_data_list[2].resample("5min").ffill()  # Forward fill missing values

    features = pd.concat(raw_data_list, axis=1, join="inner")
    targets = targets.iloc[29:-11]

    print(features)
    print(targets)

    params = HyperParams(name="lstm", sequential=True, n_features=10, n_targets=5, epochs=250)
    _, datasplit_raw_features, _, _, _, datasets = datapipe(features, targets, params)


    datasets_list.append(datasets)


    lstmmtf = LSTMMTF(params)
    #lstmmtf.train_model(datasets)
    lstmmtf.test_model(datasets)


    return None

    backtest_df = pd.DataFrame()
    backtest_df["close"] = datasplit_raw_features.test.features["close"].iloc[14:, :1]
    backtest_df["pred"] = torch.argmax(lstmmtf.model.forward(datasets.test.features.to("cuda")), dim=1).cpu().detach().numpy()
    backtest_df["pred-1"] = backtest_df["pred"].shift(1)
    backtest_df["pred-2"] = backtest_df["pred"].shift(2)

    print(backtest_df)
    
    df = backtest_df



    # Initialize variables
    position = 0  # Track the bot's position: 0 = no position, 1 = holding a position
    entry_price = 0  # Price when the bot bought the stock
    cash = 0  # Cash balance (for simplicity)
    returns = []  # Store the returns
    buy_signals = []  # Store buy signal timestamps
    sell_signals = []  # Store sell signal timestamps

    # Loop through each row
    for i, row in df.iterrows():
        if (row['pred'] == 4) and position == 0:  # Buy signal and no position
            position = 1
            entry_price = row['close'] * 1.002 # Record the entry price + costs
            buy_signals.append(row.name)  # Store buy signal timestamp (index)
        elif (entry_price * 0.98> row["close"] or entry_price * 1.04 < row["close"]) and position == 1:  # Sell signal and bot is holding
            position = 0
            cash += row['close'] - entry_price  # Calculate return from the trade
            sell_signals.append(row.name)  # Store sell signal timestamp (index)

        # If position is still held, no transaction
        returns.append(cash)

    print(len(buy_signals))
    # Add the cumulative returns to the DataFrame
    df['returns'] = returns

    # Plot price and return
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the stock price
    ax1.plot(df.index, df['close'], label='Stock Price', color='blue', alpha=0.7)
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Stock Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Mark buy and sell signals
    ax1.scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='green', label='Buy Signal', zorder=5)
    ax1.scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='red', label='Sell Signal', zorder=5)

    # Create a second y-axis to plot returns
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['returns'], label='Cumulative Returns', color='orange', alpha=0.7)
    ax2.set_ylabel('Cumulative Returns', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add a legend
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    # Show the plot
    plt.title('Stock Price, Buy/Sell Signals, and Cumulative Returns')
    plt.tight_layout()
    plt.show()



def init():
    dbm = DB_Manager()
    tables = []
    tables.append(dbm.db.table_name("BTC/USDT", "5m"))
    tables.append(dbm.db.table_name("BTC/USDT", "15m"))
    tables.append(dbm.db.table_name("BTC/USDT", "1h"))

    return [dbm.db.select_data(table_name) for table_name in tables]


if __name__ == "__main__":
    test()


