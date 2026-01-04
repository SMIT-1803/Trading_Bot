import pandas as pd
from pandas import DataFrame
import numpy as np

def add_daily_ema_bias(df_1d: DataFrame, ema_span: int = 200) -> DataFrame:
    out = df_1d.copy()
    out["ema200"] = out["close"].ewm(span=ema_span, adjust=False).mean()
    out["long_bias_1d"] = out["close"] > out["ema200"]
    return out


def add_trend_ok_to_execution(df_4h: DataFrame, df_1d: DataFrame) -> DataFrame:
    """
    Map 1D long bias onto 4H execution candles without lookahead.

    Rule:
      - Compute daily bias (long_bias_1d) on 1D candles.
      - Shift by 1 day so that day D uses bias from day D-1 (daily candle must be closed).
      - For each 4H timestamp, use the most recent available shifted daily bias at or before that time.
    """
    out_4h = df_4h.copy()

    out_4h.sort_index(inplace=True)
    d1 = df_1d.sort_index()

    if "long_bias_1d" not in d1.columns:
        raise RuntimeError("TREND ERROR: df_1d_bias must contain 'long_bias_1d'.")

    bias_prev = d1["long_bias_1d"].shift(1)

    trend_ok = bias_prev.reindex(out_4h.index, method="ffill")

    out_4h["trend_ok"] = trend_ok
    return out_4h


def add_can_trade(df_4h: DataFrame) -> DataFrame:
    """
    Requires:
      - can_enter (from volatility)
      - trend_ok
    """
    out = df_4h.copy()
    if "can_enter" not in out.columns:
        raise RuntimeError("PERMISSION ERROR: Missing 'can_enter'.")
    if "trend_ok" not in out.columns:
        raise RuntimeError("PERMISSION ERROR: Missing 'trend_ok'.")

    out["can_trade"] = out["can_enter"].fillna(False) & out["trend_ok"].fillna(False)
    return out

################### Checks ########################
# from src.data_layer.storage import load_ohlcv
# from src.features.volatility import add_volatility_features
# df_1d = load_ohlcv("kraken","BTC/USDT","1d","data/raw")
# df_4h = load_ohlcv("kraken","BTC/USDT","4h","data/raw")
# out_4h = add_volatility_features(df_4h)
# out = add_daily_ema_bias(df_1d)
# out_4h = add_trend_ok_to_execution(out_4h, out)
# final_4h = add_can_trade(out_4h)
# print(final_4h)