import sqlite3 as sql
import pandas as pd

class Database():
    def __init__(self):
        self.conn = sql.connect('database/stockprices.db')
        self.cursor = self.conn.cursor()

    def create_ohlcv_table(self, table_name):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                timestamp DATETIME PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )''')
        return None
    
    def insert_data(self, ticker, timeframe, data) -> None:
        table_name = self.table_name(ticker, timeframe)
        
        self.create_ohlcv_table(table_name)

        self.cursor.executemany(f'''
            INSERT OR IGNORE INTO "{table_name}" 
            (timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)''', data)

        self.conn.commit()

    def select_data(self, table_name, start=0, end=1e100) -> pd.DataFrame:
        """Fetches data in table_name from start to (inclusive) end."""
        self.create_ohlcv_table(table_name)

        query = f'SELECT * FROM "{table_name}" WHERE timestamp >= ? AND timestamp <= ?'
        self.cursor.execute(query, (start, end))

        rows = self.cursor.fetchall()
        columns = [desc[0] for desc in self.cursor.description]

        return pd.DataFrame(rows, columns=columns)
    
    def delete_data(self, table_name, end, start=0) -> None:
        query = f'DELETE FROM "{table_name}" WHERE timestamp >= ? AND timestamp <= ?'

        try:
            self.cursor.execute(query, (start, end))
        except Exception as e:
            raise Exception(f"Failed to delete data between {start} and {end}: {e}")
        
        return None

    def table_name(self, ticker, timeframe) -> str:
        ticker = str.replace(ticker, "/", "_")
        return ticker + "_" + timeframe
    

if __name__ == "__main__":
    db = Database()
    print(db.select_data("BTC_USDT_1s"))
    
    #db.cursor.execute('DROP TABLE "BTC/USDT"')