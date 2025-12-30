import pandas as pd
import numpy as np

def validate_ohlcv(df, timeframe):
    schema_validator(df)
    df = timestamp_validator(df)
    candle_integrity_validator(df)
    timeframe_sanity_validator(df, timeframe)
    return df


# the required columns exist
# required columns are numeric (except timestamp)
# no missing values in required columns
def schema_validator(df):
    required_columns = {"open", "high", "low", "close", "volume"}
    is_valid = required_columns.issubset(df.columns)
    if not is_valid:
        missing = required_columns - set(df.columns)
        raise RuntimeError(f"SCHEMA ERROR: Missing required columns: {missing}")

    for col in required_columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        missing_values_count = df[col].isna().sum()
        if not is_num:
            raise RuntimeError(
                f"DATATYPE ERROR: Column '{col}' must be numeric (float/int). Found {df[col].dtype}"
            )
        if missing_values_count > 0:
            raise RuntimeError(
                f"DATA GAP: Column '{col}' contains {missing_values_count} missing (NaN) values."
            )

# timestamp indes is datetime
# timestamps can be parsed
# timestamps are UTC
# no duplicates
# strictly increasing
def timestamp_validator(df):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(
            "INDEX ERROR: DataFrame index must be a pandas DatetimeIndex."
        )

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif str(df.index.tz) != "UTC":
        df.index = df.index.tz_convert("UTC")

    if df.index.has_duplicates:
        dup_ts = df.index[df.index.duplicated()].unique()[:5]
        raise RuntimeError(f"DUPLICATE ERROR: Found duplicate timestamps, {dup_ts}")

    if not df.index.is_monotonic_increasing:
        raise RuntimeError(
            "CHRONOLOGY ERROR: Timestamps are not in strictly increasing order. Try df.sort_index()."
        )
    return df



def candle_integrity_validator(df):
    cond1 = df["high"] >= df["low"]

    cond2 = (df["open"] <= df["high"]) & (df["open"] >= df["low"])
    cond3 = (df["close"] <= df["high"]) & (df["close"] >= df["low"])
    cond4 = df["volume"] >= 0
    cond5 = (
        (df["open"] >= 0) & (df["high"] >= 0) & (df["low"] >= 0) & (df["close"] >= 0)
    )

    is_valid = cond1 & cond2 & cond3 & cond4 & cond5

    if not is_valid.all():
        invalid_count = (~is_valid).sum()
        first_bad = df.index[~is_valid][0]
        raise RuntimeError(
            f"INTEGRITY ERROR: {invalid_count} candles have impossible price/volume values. First failure at: {first_bad}"
        )

def timeframe_sanity_validator(df, timeframe):
    if len(df) > 50:
        tf = timeframe.lower().strip()
        tf_map = {
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
            "1day": pd.Timedelta(days=1),
            "d": pd.Timedelta(days=1),
        }

        if tf not in tf_map:
            raise RuntimeError(
                f"INPUT ERROR: Unsupported timeframe '{timeframe}'. Use '4h' or '1d'."
            )

        expected_tf = tf_map[tf]
        tol = pd.Timedelta(minutes=5)
        min_match_ratio = 0.8

        deltas = df.index.to_series().diff().iloc[1:]
        matches = (deltas - expected_tf).abs() <= tol
        if len(matches) > 0:
            average_score = matches.mean()
            match_ratio = float(average_score)
        else:
            match_ratio = 1.0
        if match_ratio < min_match_ratio:
            raise RuntimeError(
                f"TIMEFRAME ERROR: Only {match_ratio} of consecutive gaps match. Timeframe spacing inconsistent."
            )


# Tests
# data = [
#     {"Timestamp": 1714521600000, "open": 145.20, "high": 148.50, "low": 144.10, "close": 147.40, "volume": 120000},
#     {"Timestamp": 1714536000000, "open": 147.40, "high": 150.80, "low": 146.35, "close": 149.75, "volume": 95000},
#     {"Timestamp": 1714550400000, "open": 149.75, "high": 152.10, "low": 148.70, "close": 151.00, "volume": 110000},
#     {"Timestamp": 1714564800000, "open": 151.00, "high": 151.05, "low": 145.80, "close": 146.85, "volume": 180000},
#     {"Timestamp": 1714579200000, "open": 146.85, "high": 147.95, "low": 142.50, "close": 143.60, "volume": 150000},
#     {"Timestamp": 1714593600000, "open": 143.60, "high": 145.70, "low": 141.40, "close": 144.45, "volume": 70000},
#     {"Timestamp": 1714608000000, "open": 144.45, "high": 146.55, "low": 143.30, "close": 145.40, "volume": 60000},
#     {"Timestamp": 1714622400000, "open": 145.40, "high": 148.60, "low": 144.35, "close": 147.55, "volume": 90000},
#     {"Timestamp": 1714636800000, "open": 147.55, "high": 149.90, "low": 146.55, "close": 148.80, "volume": 130000},
#     {"Timestamp": 1714651200000, "open": 148.80, "high": 152.20, "low": 147.75, "close": 151.15, "volume": 200000},
# ]

# df = pd.DataFrame(data)
# df = df.rename(columns={"Timestamp":"timestamp"})
# df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
# df.set_index("timestamp", inplace=True)
# print(df)
# validate_ohlcv(df, "4h")

