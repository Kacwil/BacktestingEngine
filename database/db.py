import sqlite3 as sql
import pandas as pd
from utils.dataclasses import Data

class Database():
    def __init__(self):
        self.conn = sql.connect('database/stockprices.db')
        self.cursor = self.conn.cursor()
        self.update_table_data()
        print(f"Connected to stockprices.db, available tables: {self.table_data.keys()}")


        reset_feature_target = False
        if reset_feature_target == True:
            self.cursor.execute(f"DROP TABLE BTC_USDT_1s_features")
            self.cursor.execute(f"DROP TABLE BTC_USDT_1s_targets")

    def update_table_data(self):
        tables = self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = [table[0] for table in tables]
        table_data = {}

        for table in tables:
            self.cursor.execute(f'SELECT timestamp FROM "{table}" ORDER BY timestamp ASC LIMIT 1')
            earliest = self.cursor.fetchone()

            self.cursor.execute(f'SELECT timestamp FROM "{table}" ORDER BY timestamp DESC LIMIT 1')
            latest = self.cursor.fetchone()

            if earliest is not None and latest is not None:
                table_data[table] = {"earliest": earliest[0], "latest": latest[0]}
            else:
                table_data[table] = {"earliest": None, "latest": None}

        self.table_data = table_data
        return None

    def create_ohlcv_table(self, name):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{name}" (
                timestamp DATETIME PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )''')
        return True


    def table_exists_in_db(self, name):
        if name in self.table_data:
            return True
        raise ValueError(f"Table {name} doesn't exists in the database!")
        
    def insert_ohlcv_data(self, ticker, timeframe, data):
        table = self.table_name(ticker, timeframe)
        
        # 1: Create table if there is none
        self.create_ohlcv_table(table)

        # 2: Insert data
        self.cursor.executemany(f'''
            INSERT OR IGNORE INTO "{table}" 
            (timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)''', data)

        self.conn.commit()
        self.update_table_data()

    def select_ohlcv_data(self, ticker, timeframe, start=None, end=None, limit=None, descending=False):
        table = self.table_name(ticker, timeframe)
        return self._select_data(table, start, end, limit, descending)
    
    def select_feature_target_data(self, ticker, timeframe, start=None, end=None, limit=None, descending=False):

        name_features = self.table_name(ticker, timeframe, "_features")
        features = self._select_data(name_features, start, end, limit, descending)

        name_targets = self.table_name(ticker, timeframe, "_targets")
        targets = self._select_data(name_targets, start, end, limit, descending)

        return Data(features, targets)

    def _select_data(self, table, start=None, end=None, limit=None, descending=False):
        """ Fetches data from the database and returns a dataframe """

        if self.table_exists_in_db(table):

            query = f'SELECT * FROM "{table}"'
            conditions = []
            params = []

            if start:
                conditions.append("timestamp >= ?")
                params.append(start)

            if end:
                conditions.append("timestamp <= ?")
                params.append(end)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            order = "DESC" if descending else "ASC"
            query += f" ORDER BY timestamp {order}"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]

            return pd.DataFrame(rows, columns=columns)
        


    def insert_feature_or_target_data(self, table_name, df):

        if df.empty:
            return

        # 1: Create table if there is none
        col_defs = []
        for col, dtype in df.dtypes.items():
            if "int" in str(dtype):
                sql_type = "INTEGER"
            elif "float" in str(dtype):
                sql_type = "REAL"
            elif "bool" in str(dtype):
                sql_type = "BOOLEAN"
            else:
                sql_type = "TEXT"

            if col == "timestamp":
                col_defs.append(f'"{col}" {sql_type} PRIMARY KEY')
            else:
                col_defs.append(f'"{col}" {sql_type}')
        
        columns_sql = ", ".join(col_defs)
        create_stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'
        self.cursor.execute(create_stmt)

        # 2: Insert Data
        columns = df.columns.tolist()
        placeholders = ', '.join(['?'] * len(columns))
        column_names = ', '.join(f'"{col}"' for col in columns)

        insert_stmt = f'''INSERT OR REPLACE INTO "{table_name}" ({column_names}) VALUES ({placeholders})'''
        self.cursor.executemany(insert_stmt, df.itertuples(index=False, name=None))
        self.conn.commit()

        self.update_table_data()


    def table_name(self, ticker, timeframe, table_type=""):
        ticker = str.replace(ticker, "/", "_")
        return ticker + "_" + timeframe + table_type
    


db = Database()