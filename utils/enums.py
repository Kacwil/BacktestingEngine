from enum import Enum

class TICKERS(str, Enum):
    BTC_USDT = "BTC/USDT"
    ETH_USDT = "ETH/USDT"
    DOGE_USDT = "DOGE/USDT"

    def __str__(self):
        return self.value

class TIMEFRAMES(str, Enum):
    S = "1s"
    M = "1m"
    M15 = "15m"
    H = "1h"
    D = "1d"

    def __str__(self):
        return self.value

class MSG_COMMANDS(str, Enum):
    SELECT_OHLCV = "select_ohlcv"
    SELECT_FEATURES_TARGETS = "select_features_targets"
    SHUTDOWN = "shutdown"
    PING = "ping"
    UPDATE_TO_LATEST = "update_to_latest"
    GET_TABLES = "get_tables"
    CREATE_TABLE = "create_table"

    def __str__(self):
        return self.value
    
    
