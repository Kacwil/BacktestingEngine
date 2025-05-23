import datetime

from database.db import Database
from database.api_fetcher import BinanceFetcher
from database.feature_extractor import Feature_Extractor
from utils.utils import timeframe_to_timeskip
from utils.enums import TABLE_TYPE
import asyncio

class DB_Manager():
    def __init__(self):
        self.db = Database()
        self.fetcher = BinanceFetcher()
        self.extractor = Feature_Extractor()

    async def populate_data(self, ticker, timeframe, start=None, end=None, backfill_n_rows=None):
        table = self.db.table_name(ticker, timeframe)
        time_skip = timeframe_to_timeskip(timeframe)

        if end is None:
            end = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)

        if start is None:
            start = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000) - (1000 * 60 * 60 * 24)

        earliest = self.db.table_data[table]["earliest"]
        latest = self.db.table_data[table]["latest"]

        async def fetch_and_insert(fetch_start, fetch_end):
            async for candles in self.fetcher.fetch_ohlcv_stream(ticker, timeframe, fetch_start, fetch_end):
                self.db.insert_ohlcv_data(ticker, timeframe, candles)
                print(f"[DB] Inserted {len(candles)} rows → {ticker} [{timeframe}] | {fetch_start} → {fetch_end}")

        if backfill_n_rows != None:

            if earliest is None:
                raise ValueError("The table is probably empty — no data to backfill from.")
            
            start = earliest - (backfill_n_rows * time_skip)
            end = earliest - time_skip

            await fetch_and_insert(start, end)
            return

        if earliest == None and latest == None:
            await fetch_and_insert(start, end)
            return

        if start >= earliest and end <= latest:
            print("Data (should) already exist in the requested range.")
            return

        if start < earliest:
            await fetch_and_insert(start, earliest - time_skip)

        if end > latest:
            await fetch_and_insert(latest + time_skip, end)

    async def validate_table_data(self, ticker, timeframe):
        table = self.db.table_name(ticker, timeframe)
        print(f"Validating {table}")

        data = self.db.select_ohlcv_data(ticker, timeframe)
        if not data:
            print(f"No data found in {table}")
            return

        skip = timeframe_to_timeskip(timeframe)
        expected_time = data[0][0]

        for actual_time, *_ in data:
            if actual_time != expected_time:
                await self._handle_missing_gap(ticker, timeframe, expected_time, actual_time)
                expected_time = actual_time + skip
            else:
                expected_time += skip
        print("Data validation complete")

    async def _handle_missing_gap(self, ticker, timeframe, start, end):
        print(f"Missing data at {start}, next data exists at {end}")
        async for candles in self.fetcher.fetch_ohlcv_stream(ticker, timeframe, start, end):
            self.db.insert_ohlcv_data(ticker, timeframe, candles)


    async def extract_features(self, ticker, timeframe, start=None, end=None):
        await self.validate_table_data(ticker, timeframe)

        await asyncio.sleep(0)

        table_features = self.db.table_name(ticker, timeframe, TABLE_TYPE.FEATURES)
        table_targets = self.db.table_name(ticker, timeframe, TABLE_TYPE.TARGETS)
        earliest, latest = end, end

        if self.db.table_data.get(table_features):
            earliest = self.db.table_data[table_features]["earliest"]
            latest = self.db.table_data[table_features]["latest"]
            data1 = self.db.select_ohlcv_data(ticker, timeframe, start, earliest)
            data2 = self.db.select_ohlcv_data(ticker, timeframe, latest - 3500*1000, end)
            data = data1 + data2
        else:
            data = self.db.select_ohlcv_data(ticker, timeframe, start, end)

        df_features, df_targets = self.extractor.create_features_targets(data)

        await asyncio.sleep(0)

        print(f"[DB_Manager] trying to insert {len(df_features)} features and targets")

        self.db.insert_feature_or_target_data(table_features, df_features)
        self.db.insert_feature_or_target_data(table_targets, df_targets)
