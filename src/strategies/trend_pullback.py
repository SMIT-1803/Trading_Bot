import pandas as pd
import numpy as np
from pandas import DataFrame


def add_trend_pullback_indicators(
    df_4h: DataFrame, ema_fast: int = 50, ema_slow: int = 200
) -> DataFrame:
    df = df_4h.copy()
    df.sort_index(inplace=True)
    df["ema_fast_4h"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow_4h"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    df["trend_4h_ok"] = (
        df["ema_fast_4h"] > df["ema_slow_4h"]
    )  # Shows sustained upward momentum

    cond = (df["low"] <= df["ema_fast_4h"]) & (
        df["close"] > df["ema_fast_4h"]
    )  # This condition ensures pullback up was successfull after the low <= ema_fast but close > ema_fast
    df["pullback_reclaim_ema_fast"] = cond

    df["bullish_candle"] = (
        df["close"] > df["open"]
    )  # To enter when the momentum shifts.

    return df


def add_entry_signals(df_4h: DataFrame) -> DataFrame:
    df = df_4h.copy()
    entry_long_cond = (
        df["can_trade"]
        & df["trend_4h_ok"]
        & df["pullback_reclaim_ema_fast"]
        & df["bullish_candle"]
    )
    df["entry_long"] = entry_long_cond.fillna(False)

    df["entry_price"] = np.nan
    entry_price_cond = df["entry_long"]
    df.loc[entry_price_cond, "entry_price"] = df["close"]

    return df


def add_initial_stop_and_risk(df_4h:DataFrame, k:float=1.5)->DataFrame:
    df = df_4h.copy()
    df["stop_price"] = np.nan
    df.loc[df["entry_long"],"stop_price"] = df["entry_price"] - k * df["atr"]

    df["risk_per_unit"] = np.nan
    df.loc[df["entry_long"],"risk_per_unit"] = df["entry_price"] - df["stop_price"]

    # Stop distance is same as risk per unit, but for debugging
    df["stop_distance"] = np.nan
    df.loc[df["entry_long"],"stop_distance"] = df["entry_price"] - df["stop_price"]
    return df


###################### TESTS #####################
# from src.data_layer.storage import load_ohlcv
# from src.features.volatility import add_volatility_features
# from src.features.trend import (
#     add_daily_ema_bias,
#     add_can_trade,
#     add_trend_ok_to_execution,
# )

# df_1d = load_ohlcv("kraken", "BTC/USD", "1d", "data/raw")
# df_1d = add_daily_ema_bias(df_1d)
# df_4h = load_ohlcv("kraken", "BTC/USD", "4h", "data/raw")

# df_4h = add_volatility_features(df_4h)
# df_4h = add_trend_ok_to_execution(df_4h, df_1d)
# df_4h = add_can_trade(df_4h)

# df_4h = add_trend_pullback_indicators(df_4h)
# df_4h = add_entry_signals(df_4h)

# print((df_4h["entry_long"] & ~df_4h["can_trade"]).any())
# print(
#     (
#         df_4h.loc[df_4h["entry_long"], "entry_price"]
#         != df_4h.loc[df_4h["entry_long"], "close"]
#     ).any()
# )
# print(df_4h["entry_long"].sum())
