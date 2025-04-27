import ccxt
import asyncio
from utils.utils import timeframe_to_timeskip

class BinanceFetcher:
    """
    Fetches the ohlcv data for ticker-timeframe from timestamps start to end.
    """

    def __init__(self):
        self.exchange = ccxt.binance()
    
    async def fetch_ohlcv_stream(self, symbol, timeframe, start, end, delay=0.051, max_retries=3):
        loop = asyncio.get_running_loop()
        retries = 0

        while start < end:
            try:
                candles = await loop.run_in_executor(None, self.exchange.fetch_ohlcv, symbol, timeframe, start, 1000)
                if not candles:
                    break
                yield candles # send candles to manager

                start = candles[-1][0] + timeframe_to_timeskip(timeframe)
                retries = 0 
                await asyncio.sleep(delay)

            except Exception as e:
                print(f"Error fetching candles: {e}")
                retries += 1
                if retries > max_retries:
                    print("Max retries exceeded.")
                    break
                backoff = delay * (2 ** retries)
                print(f"Retrying in {backoff:.2f} seconds...")
                await asyncio.sleep(backoff)
