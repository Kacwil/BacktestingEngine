from database.db import Database
from database.api_fetcher import BinanceFetcher
from utils.utils import timeframe_to_timeskip, ts_now, ts_minus7d
from utils.dataclasses import DatabaseData, TableData
import asyncio

class DB_Manager():
    def __init__(self):
        self.db = Database()
        self.db_data:DatabaseData = self.get_database_data()
        self.fetcher = BinanceFetcher()

    def get_database_data(self) -> DatabaseData:
        """Gets names, first/last timestamp and total/expected rows of each table"""

        tables = self.db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names:list[str] = [table[0] for table in tables]
        db_data = DatabaseData({})

        def get_scalar(query: str) -> int:
            self.db.cursor.execute(query)
            row = self.db.cursor.fetchone()
            return row[0] if row else 0

        for table_name in table_names:

            table_data = TableData(table_name)

            first_ts = get_scalar(f'SELECT timestamp FROM "{table_name}" ORDER BY timestamp ASC LIMIT 1')
            last_ts = get_scalar(f'SELECT timestamp FROM "{table_name}" ORDER BY timestamp DESC LIMIT 1')
            total_rows = get_scalar(f'SELECT COUNT(DISTINCT timestamp) FROM "{table_name}"')
            expected_rows = int((last_ts - first_ts)/timeframe_to_timeskip(table_name.split("_")[2])) + 1

            table_data.first = first_ts
            table_data.last = last_ts
            table_data.total_rows = total_rows
            table_data.expected_rows = expected_rows

            db_data.tables[table_name] = table_data

        return db_data

    async def populate_data(self, ticker:str, timeframe:str, start:int=None, end:int=None):
        """
        Fetch new data into the database
        First from start -> earliest_ts, then from latest_ts -> end
        """

        table_name = self.db.table_name(ticker, timeframe)
        time_skip = timeframe_to_timeskip(timeframe)
        
        self._validate_table_in_db_data(table_name)
        start, end = self._default_input_times(start, end)
        earliest = self.db_data.tables[table_name].first
        latest = self.db_data.tables[table_name].last
        total_added = 0

        if start < earliest:
            total_added += await self._fetch_and_insert(ticker, timeframe, start, earliest - time_skip)

        if end > latest:
            total_added += await self._fetch_and_insert(ticker, timeframe, latest + time_skip, end)

        self.db_data = self.get_database_data()

        return None
    
    def _default_input_times(self, start, end):
        return start or ts_minus7d(), end or ts_now()
    
    def _validate_table_in_db_data(self, table_name):
        if table_name not in self.db_data.tables:
            raise ValueError(f"Table {table_name} not in db_data.")
        return None

    async def _fetch_and_insert(self, ticker, timeframe, fetch_start, fetch_end):
        counter = 0
        async for candles in self.fetcher.fetch_ohlcv_stream(ticker, timeframe, fetch_start, fetch_end):
            self.db.insert_data(ticker, timeframe, candles)
            print(f"[DB] Inserted {len(candles)} rows → {ticker} [{timeframe}] | {fetch_start} → {fetch_end}")
            counter += len(candles)
        return counter


    async def validate_table_data(self, ticker, timeframe):
        """Checks the ohlcv dataframe for missing timestamps, and fetches the missing entries."""
        table_name = self.db.table_name(ticker, timeframe)
        data = self.db.select_data(table_name)

        if self.db_data.tables[table_name].expected_rows - self.db_data.tables[table_name].total_rows == 0:
            print("Checksum success.")
            return None

        if data.empty:
            raise Exception("No data to validate")
        
        print(self.db_data.tables[table_name].expected_rows - self.db_data.tables[table_name].total_rows)

        time_skip = timeframe_to_timeskip(timeframe)
        time = data["timestamp"][:-1]
        next_time = time.shift(-1)[:-1]

        actual_deltas = next_time - time
        gap_mask = actual_deltas > time_skip
        gap_starts = time[gap_mask]
        gap_ends = next_time[gap_mask]

        for expected, actual in zip(gap_starts, gap_ends):
            await self._fetch_and_insert(ticker, timeframe, expected + time_skip, actual)
        
        self.db_data = self.get_database_data()
        return None


if __name__ == "__main__":
    dbm = DB_Manager()

    import time

    start = time.perf_counter()
    dbm.get_database_data()
    end = time.perf_counter()
    print(end-start)
    print(dbm.db_data)

    asyncio.run(dbm.validate_table_data("BTC/USDT", "1s"))
