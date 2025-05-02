from database.db_manager import DB_Manager
from datapipe.datapipe import datapipe
from datapipe.CustomDataset import CustomDataset
from utils.dataclasses import Data, DataSplit
from models.LSTM import LSTM
from utils.HyperParameters import HyperParams
from models.ModelBase import Model_Base
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
import random
import math

@dataclass
class Params():
   init_cash = 100000
   init_position = 0

   n_actions = 3
   
class CustomEnv(gym.Env):
    """Custom Backtester Environment based on gym interface"""

    def __init__(self, params: Params, data: np.ndarray):
        self._agent_cash = params.init_cash
        self._agent_position = params.init_position
        self.data = data
        self.size = len(data)

        self.observation_space = gym.spaces.Dict({
            "price": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.action_space = gym.spaces.Dict({
            "action": gym.spaces.Discrete(2),  # 0 -> Hold, 1 -> Buy/Sell
            "qty": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)  # Greater than 0 for buy, Less than 0 for sell
        })

    def reset(self):

        self._agent_cash = random.randint(90000, 1100000)
        self._agent_position = random.uniform(-10, 10)
        self.timestep = random.randrange(0, self.size - 1000)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
  
    def _get_obs(self):
        return self.data[self.timestep] 
    
    def _get_info(self):
        return None
  
    def step(self, action):

        if action["action"] == 1:
            price = self._get_obs()
            self._trade(price, action["qty"])

    def _trade(self, price, qty):
        position = self._agent_position
        cash = self._agent_cash

        def scale_qty(qty, price, cash):
            if abs(price * qty) > cash:
                qty = (cash/price) * np.sign(qty)
            return qty

        if (position >= 0 and qty >= 0) or (position <= 0 and qty <= 0):
            qty = scale_qty(qty, price, cash)
            self._agent_cash -= price * abs(qty)
            self._agent_position += qty

        else:
            if abs(qty) <= abs(position):
                self._agent_cash += price * abs(qty)
                self._agent_position += qty

            else:
                self._agent_cash += price * abs(position)
                qty = scale_qty(qty-position, price, cash)
                self._agent_cash -= price * abs(qty)
                self._agent_position += qty






        pass















def backtest():
    data = init()
    print(data)



def init():
    dbm = DB_Manager()
    table_name = dbm.db.table_name("BTC/USDT", "1s")
    return dbm.db.select_data(table_name)

if __name__ == "__main__":
    backtest()