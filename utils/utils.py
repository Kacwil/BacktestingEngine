from datetime import datetime, timezone
import json

#---Database helper functions---
def timeframe_to_timeskip(timeframe:str):
    """Takes in a timeframe string looking like "15m" / "1s" etc and returns the miliseconds between each step """
    if timeframe == "1s": return 1000
    elif timeframe == "5s": return 1000 * 5
    elif timeframe == "15s": return 1000 * 15
    elif timeframe == "30s": return 1000 * 30
    elif timeframe == "1m": return 1000 * 60
    elif timeframe == "2m": return 1000 * 60 * 2
    elif timeframe == "3m": return 1000 * 60 * 3
    elif timeframe == "5m": return 1000 * 60 * 5
    elif timeframe == "15m": return 1000 * 60 * 15
    elif timeframe == "30m": return 1000 * 60 * 30
    elif timeframe == "1h": return 1000 * 60 * 60
    elif timeframe == "4h": return 1000 * 60 * 60 * 4
    elif timeframe == "12h": return 1000 * 60 * 60 * 12
    elif timeframe == "1d": return 1000 * 60 * 60 * 24
    else: raise ValueError(f"Unsupported timeframe: {timeframe}")

def seconds_since_newday(t):
    dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
    return 2*((dt.hour * 3600 + dt.minute * 60 + dt.second) / (60*60*24)) - 1

def seconds_since_newweek(t):
    dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
    return 2*((dt.weekday() * 86400 + seconds_since_newday(t)) / (60*60*24*7)) - 1

def ts_now():
    return int(datetime.now(timezone.utc).timestamp() * 1000)

def ts_minus7d():
    return ts_now() - (1000 * 60 * 60 * 24 * 7)

def ts_minus30d():
    return ts_now() - (1000 * 60 * 60 * 24 * 30)

def ts_minus365d():
    return ts_now() - (1000 * 60 * 60 * 24 * 365)

#---Server-client communication---
def create_msg(cmd, args=[]):
    msg = {"cmd":cmd,
           "args": args}
    return json.dumps(msg)

def read_msg(msg):
    data = json.loads(msg)
    cmd = data["cmd"]
    args = data.get("args", [])
    return cmd, args
