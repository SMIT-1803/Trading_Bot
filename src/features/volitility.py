import pandas as pd
from pandas import DataFrame
import numpy as np
from src.data_layer.validators import validate_ohlcv
from src.data_layer.storage import load_ohlcv

def get_normalized_atr(ohlcv:DataFrame, window:int=14)->DataFrame:
    df = ohlcv.copy()
    tr = np.maximum.reduce([df["high"]-df["low"],np.abs(df["high"]-df["close"].shift()), np.abs(df["low"]-df["close"].shift())])

    df["tr"] = tr

    df["atr"] = df["tr"].rolling(window=window).mean()

    df["norm_atr"] = df["atr"]/df["close"]

    return df

######################### TESTS ######################
# df = load_ohlcv("kraken", "BTC/USDT", "4h", "data/raw")
# df_norm = get_normalized_atr(df, 14)
# print(df_norm)
