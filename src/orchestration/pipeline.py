from pandas import DataFrame
import pandas as pd

from src.features.trend import (
    add_daily_ema_bias,
    add_trend_ok_to_execution,
    add_can_trade,
)
from src.features.volatility import add_volatility_features
from src.strategies.trend_pullback import (
    add_trend_pullback_indicators,
    add_entry_signals,
    add_initial_stop_and_risk,
)


def build_features_1d(df_1d: DataFrame, *, ema_span: int = 200) -> DataFrame:
    df_1d = add_daily_ema_bias(df_1d, ema_span)
    return df_1d


def build_features_4h_base(
    df_4h: DataFrame,
    *,
    atr_window: int = 14,
    pct_window: int = 360,
    thresholds: tuple = (0.35, 0.65),
) -> DataFrame:
    df_4h = add_volatility_features(df_4h, atr_window, pct_window, thresholds)
    return df_4h


def add_permissions(df_4h: DataFrame, df_1d: DataFrame) -> DataFrame:
    df_4h = add_trend_ok_to_execution(df_4h, df_1d)
    df_4h = add_can_trade(df_4h)
    return df_4h


def add_strategy_signals(
    df_4h: DataFrame, *, ema_fast: int = 50, ema_slow: int = 200, k_stop: float = 1.5
) -> DataFrame:
    df_4h = add_trend_pullback_indicators(df_4h, ema_fast, ema_slow)
    df_4h = add_entry_signals(df_4h)
    df_4h = add_initial_stop_and_risk(df_4h, k=k_stop)
    return df_4h


def build_execution_frame(
    df_4h: DataFrame,
    df_1d: DataFrame,
    *,
    ema_span_1d: int = 200,
    atr_window: int = 14,
    pct_window: int = 360,
    thresholds: tuple = (0.35, 0.65),
    ema_fast: int = 50,
    ema_slow: int = 200,
    k_stop: float = 1.5,
) -> DataFrame:
    d4 = df_4h.copy()
    d1 = df_1d.copy()

    # Basic sanity (lightweight, not full validation)
    if not isinstance(d4.index, pd.DatetimeIndex) or not isinstance(
        d1.index, pd.DatetimeIndex
    ):
        raise RuntimeError("PIPELINE ERROR: df_4h and df_1d must have a DatetimeIndex.")
    if d4.index.tz is None or d1.index.tz is None:
        raise RuntimeError("PIPELINE ERROR: indices must be timezone-aware (UTC).")

    d4.sort_index(inplace=True)
    d1.sort_index(inplace=True)

    d1_feat = build_features_1d(d1, ema_span=ema_span_1d)

    d4_feat = build_features_4h_base(
        d4,
        atr_window=atr_window,
        pct_window=pct_window,
        thresholds=thresholds,
    )

    d4_perm = add_permissions(d4_feat, d1_feat)

    d4_final = add_strategy_signals(
        d4_perm,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        k_stop=k_stop,
    )

    d4_final.sort_index(inplace=True)

    required = {
        "entry_long",
        "entry_price",
        "stop_price",
        "risk_per_unit",
        "can_trade",
        "risk_multiplier",
    }
    missing = required - set(d4_final.columns)
    if missing:
        raise RuntimeError(f"PIPELINE ERROR: missing required columns {missing}")

    return d4_final
