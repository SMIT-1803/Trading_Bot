import pandas as pd
from pandas import DataFrame
import numpy as np


# Rolling percentile of the current value within the last W values
def pct_rank_last(window: np.ndarray) -> float:
    last = window[-1]
    # Handle NaNs gracefully
    if np.isnan(last):
        return np.nan
    return float(np.mean(window <= last))


def get_normalized_atr(ohlcv: DataFrame, window: int = 14) -> DataFrame:
    df = ohlcv.copy()
    tr = np.maximum.reduce(
        [
            df["high"] - df["low"],
            np.abs(df["high"] - df["close"].shift()),
            np.abs(df["low"] - df["close"].shift()),
        ]
    )

    df["tr"] = tr

    df["atr"] = df["tr"].rolling(window=window).mean()

    df["norm_atr"] = df["atr"] / df["close"]

    df.drop(columns=["tr"], inplace=True)

    return df


def add_volatility_regime(
    df: DataFrame, W: int = 360, thresholds: tuple = (0.30, 0.65)
) -> DataFrame:
    new_df = df.copy()

    new_df["vol_pct"] = (
        new_df["norm_atr"]
        .rolling(window=W, min_periods=W)
        .apply(pct_rank_last, raw=True)
    )

    low_vol_cond = new_df["vol_pct"] < thresholds[0]
    trade_vol_cond = (thresholds[0] <= new_df["vol_pct"]) & (
        new_df["vol_pct"] < thresholds[1]
    )
    high_vol_cond = new_df["vol_pct"] >= thresholds[1]

    new_df["vol_regime"] = None
    new_df.loc[low_vol_cond, "vol_regime"] = "LOW_VOL"
    new_df.loc[trade_vol_cond, "vol_regime"] = "TRADE_OK"
    new_df.loc[high_vol_cond, "vol_regime"] = "HIGH_VOL"

    new_df["can_enter"] = False
    # new_df.loc[new_df["vol_regime"] == "LOW_VOL", "can_enter"] = True
    new_df.loc[new_df["vol_regime"] == "TRADE_OK", "can_enter"] = True
    new_df.loc[new_df["vol_regime"] == "HIGH_VOL", "can_enter"] = True

    new_df["risk_multiplier"] = 0.0
    # new_df.loc[new_df["vol_regime"] == "LOW_VOL", "risk_multiplier"] = 0.2
    new_df.loc[new_df["vol_regime"] == "TRADE_OK", "risk_multiplier"] = 1.0
    new_df.loc[new_df["vol_regime"] == "HIGH_VOL", "risk_multiplier"] = 0.5

    return new_df


def add_volatility_features(
    ohlcv_4h: DataFrame,
    atr_window: int = 14,
    pct_window: int = 360,
    thresholds: tuple[float, float] = (0.35, 0.65),
) -> DataFrame:
    df = get_normalized_atr(ohlcv_4h, window=atr_window)
    df = add_volatility_regime(df, W=pct_window, thresholds=thresholds)
    return df

######################### TESTS ######################
# df = load_ohlcv("kraken", "BTC/USD", "4h", "data/raw")
# df_norm = get_normalized_atr(df, 14)
# df_norm = add_volatility_regime(df_norm)
# print(df_norm)
# print(df_norm.loc["2024-12-01 20:00:00+00:00"])
# print(df_norm.info())
