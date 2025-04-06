import asyncio
import struct
import pickle
import datetime

from .db_manager import DB_Manager
from utils.enums import MSG_COMMANDS
from utils.utils import read_msg 

class DB_Server():
    def __init__(self, address="localhost", port=6000, authkey=b"hello"):
        self.db_manager = DB_Manager()
        self.address = address
        self.port = port

        asyncio.run(self.start_server())

    async def start_server(self): 
        server = await asyncio.start_server(self.db_server, self.address, self.port)
        print(f"Listening on {self.address}:{self.port}")

        asyncio.create_task(self.fetch_data())
        asyncio.create_task(self.extract_features())

        async with server:
            await server.serve_forever()

    async def db_server(self, reader, writer):
        try:
            data = await reader.read(4096)
            msg = data.decode()
            print(f"Message recieved: {msg}")
            cmd, args = read_msg(msg)

            if cmd == MSG_COMMANDS.SHUTDOWN:
                writer.write(b"Shutting down")
                await writer.drain()
                asyncio.get_event_loop().stop()

            elif cmd == MSG_COMMANDS.PING:
                print("PING")
                data = pickle.dumps("PONG")
                await self.send_data(data, writer)

            elif cmd == MSG_COMMANDS.SELECT_OHLCV:
                result = self.db_manager.db.select_ohlcv_data(*args)
                data = pickle.dumps(result)
                await self.send_data(data, writer)

            elif cmd == MSG_COMMANDS.SELECT_FEATURES_TARGETS:
                result = self.db_manager.db.select_feature_target_data(*args)
                data = pickle.dumps(result)
                await self.send_data(data, writer)          

            
        except Exception as e:
            print("Error:", e)

        finally:
            writer.close()
            await writer.wait_closed()

    async def send_data(self, data, writer):
        """Sends the data to client. Encodes length since data might be large"""
        length = struct.pack("!I", len(data))
        writer.write(length + data)
        await writer.drain()

    async def fetch_data(self):
        while True:
            print(f"[Server] fetching data...")
            try:
                await self.db_manager.populate_data("BTC/USDT", "1s")
            except Exception as e:
                print(f"[Server] Error: {e}")
            await asyncio.sleep(0.01)


    async def extract_features(self):
        while True:
            print(f"[Server] extracting features...")
            now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
            try:
                await self.db_manager.extract_features("BTC/USDT", "1s", start = now - 1000 * 100000)
            except Exception as e:
                print(f"[Server] Error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    db_server = DB_Server()