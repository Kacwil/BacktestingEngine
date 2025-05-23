import matplotlib.pyplot as plt
from datetime import datetime, timezone
import json


#---Database helper functions---
def timeframe_to_timeskip(timeframe):
    if timeframe == "1s": return 1000
    elif timeframe == "1m": return 1000 * 60
    elif timeframe == "15m": return 1000 * 60 * 15
    elif timeframe == "1h": return 1000 * 60 * 60
    elif timeframe == "1d": return 1000 * 60 * 60 * 24
    elif timeframe == "1w": return 1000 * 60 * 60 * 24 * 7
    else: raise ValueError(f"Unsupported timeframe: {timeframe}")

def seconds_since_newday(t):
    dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
    return 2*((dt.hour * 3600 + dt.minute * 60 + dt.second) / (60*60*24)) - 1

def seconds_since_newweek(t):
    dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
    return 2*((dt.weekday() * 86400 + seconds_since_newday(t)) / (60*60*24*7)) - 1


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
