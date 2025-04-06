import asyncio
import struct
import pickle
import subprocess
import datetime
import time

from utils.enums import TICKERS, TIMEFRAMES, MSG_COMMANDS
from utils.utils import create_msg


class DB_Client():
    def __init__(self, address="localhost", port=6000):
        self.address = address
        self.port = port

    def ping(self):
        msg = create_msg(MSG_COMMANDS.PING)
        return asyncio.run(self.send_message(msg))
    
    def shutdown(self):
        msg = create_msg(MSG_COMMANDS.SHUTDOWN)
        return asyncio.run(self.send_message(msg))

    def select_ohlcv(self, ticker, timeframe, start=None, end=None, limit=None, descending=False):
        msg = create_msg(MSG_COMMANDS.SELECT_OHLCV, [ticker, timeframe, start, end, limit, descending])
        return asyncio.run(self.send_message(msg))
    

    def select_features_targets(self, ticker, timeframe, start=None, end=None, limit=None, descending=False):
        msg = create_msg(MSG_COMMANDS.SELECT_FEATURES_TARGETS, [ticker, timeframe, start, end, limit, descending])
        return asyncio.run(self.send_message(msg))
    


    async def send_message(self, message: str):
        while True:
            try:
                reader, writer = await asyncio.open_connection(self.address, self.port)
                break
            except ConnectionRefusedError:
                print("Server not found. Trying to open a new server")
                subprocess.Popen('start cmd /k "python -m database.db_server"',shell=True)
                await asyncio.sleep(1)

        #Send
        writer.write(message.encode())
        await writer.drain()

        #Read
        length_bytes = await reader.readexactly(4)
        msg_len = struct.unpack("!I", length_bytes)[0]

        payload = await reader.readexactly(msg_len)
        data = pickle.loads(payload)
        
        writer.close()
        await writer.wait_closed()
        return data
