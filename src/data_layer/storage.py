import os
import pandas as pd
from pandas import DataFrame
from src.data_layer.validators import validate_ohlcv


def symbol_to_path_component(symbol: str) -> str:
    """Convert CCXT symbol (e.g. BTC/USD) to filesystem-safe name (BTC-USD)."""
    symbol = symbol.strip().replace("/", "-")
    return symbol


def get_ohlcv_path(exchange: str, symbol: str, timeframe: str, root_dir: str) -> str:
    """Return canonical path to OHLCV CSV for given exchange/symbol/timeframe."""
    symbol = symbol_to_path_component(symbol)
    path = os.path.join(root_dir, exchange, symbol, timeframe, "ohlcv.csv")
    return path


def load_ohlcv(
    exchange: str, symbol: str, timeframe: str, root_dir: str
) -> DataFrame | None:
    path = get_ohlcv_path(exchange, symbol, timeframe, root_dir)
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"LOAD ERROR: Missing 'timestamp' column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    try:
        df = validate_ohlcv(df, timeframe)
    except RuntimeError as e:
        raise RuntimeError(f"{e} while loading OHLCV")
    return df


def save_ohlcv(
    df: DataFrame, exchange: str, symbol: str, timeframe: str, root_dir: str
) -> None:
    required_columns = ["open", "high", "low", "close", "volume"]
    path = get_ohlcv_path(exchange, symbol, timeframe, root_dir)
    try:
        df = validate_ohlcv(df, timeframe)
    except RuntimeError as e:
        raise RuntimeError(f"SAVE_OHLCV ERROR: {e}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df[required_columns].copy()
    out.insert(
        0, "timestamp", df.index.astype(str)
    )  # UTC-aware index -> ISO-like strings
    out.to_csv(path, index=False)


def merge_ohlcv(
    old_df: DataFrame | None, new_df: DataFrame, timeframe: str
) -> DataFrame:
    try:
        validate_ohlcv(new_df, timeframe)
    except RuntimeError as e:
        raise RuntimeError(f"MERGE OHLCV ERROR: {e}")
    if old_df is None:
        return new_df
    else:
        try:
            validate_ohlcv(old_df, timeframe)
        except RuntimeError as e:
            raise RuntimeError(f"MERGE OHLCV ERROR: {e}")
        merged = pd.concat([old_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")]
        merged.sort_index(inplace=True)
        try:
            validate_ohlcv(merged, timeframe)
        except RuntimeError as e:
            raise RuntimeError(f"MERGE OHLCV ERROR: {e}")
        return merged


def upsert_ohlcv(
    new_df: DataFrame, exchange: str, symbol: str, timeframe: str, root_dir: str
) -> DataFrame:
    old_df = load_ohlcv(exchange, symbol, timeframe, root_dir)
    merged = merge_ohlcv(old_df, new_df, timeframe)
    save_ohlcv(merged, exchange, symbol, timeframe, root_dir)
    return merged


######################## TESTS ###########################
# data = [
#     {
#         "Timestamp": 1714521600000,
#         "open": 145.20,
#         "high": 148.50,
#         "low": 144.10,
#         "close": 147.40,
#         "volume": 120000,
#     },
#     {
#         "Timestamp": 1714536000000,
#         "open": 147.40,
#         "high": 150.80,
#         "low": 146.35,
#         "close": 149.75,
#         "volume": 95000,
#     },
#     {
#         "Timestamp": 1714550400000,
#         "open": 149.75,
#         "high": 152.10,
#         "low": 148.70,
#         "close": 151.00,
#         "volume": 110000,
#     },
#     {
#         "Timestamp": 1714564800000,
#         "open": 151.00,
#         "high": 151.05,
#         "low": 145.80,
#         "close": 146.85,
#         "volume": 180000,
#     },
#     {
#         "Timestamp": 1714579200000,
#         "open": 146.85,
#         "high": 147.95,
#         "low": 142.50,
#         "close": 143.60,
#         "volume": 150000,
#     },
#     {
#         "Timestamp": 1714593600000,
#         "open": 143.60,
#         "high": 145.70,
#         "low": 141.40,
#         "close": 144.45,
#         "volume": 70000,
#     },
#     {
#         "Timestamp": 1714608000000,
#         "open": 144.45,
#         "high": 146.55,
#         "low": 143.30,
#         "close": 145.40,
#         "volume": 60000,
#     },
#     {
#         "Timestamp": 1714622400000,
#         "open": 145.40,
#         "high": 148.60,
#         "low": 144.35,
#         "close": 147.55,
#         "volume": 90000,
#     },
#     {
#         "Timestamp": 1714636800000,
#         "open": 147.55,
#         "high": 149.90,
#         "low": 146.55,
#         "close": 148.80,
#         "volume": 130000,
#     },
#     {
#         "Timestamp": 1714651200000,
#         "open": 148.80,
#         "high": 152.20,
#         "low": 147.75,
#         "close": 151.15,
#         "volume": 200000,
#     },
# ]

# newData = [
#     {
#         "Timestamp": 1714636800000,
#         "open": 1147.55,
#         "high": 1149.90,
#         "low": 1146.55,
#         "close": 1148.80,
#         "volume": 1130000,
#     },
#     {
#         "Timestamp": 1714651200000,
#         "open": 1148.80,
#         "high": 1152.20,
#         "low": 1147.75,
#         "close": 1151.15,
#         "volume": 1200000,
#     },
#     {
#         "Timestamp": 1714665600000,
#         "open": 151.15,
#         "high": 153.40,
#         "low": 150.90,
#         "close": 152.80,
#         "volume": 115000,
#     },
#     {
#         "Timestamp": 1714680000000,
#         "open": 152.80,
#         "high": 154.00,
#         "low": 151.50,
#         "close": 151.90,
#         "volume": 98000,
#     },
#     {
#         "Timestamp": 1714694400000,
#         "open": 151.90,
#         "high": 152.50,
#         "low": 149.80,
#         "close": 150.25,
#         "volume": 105000,
#     },
#     {
#         "Timestamp": 1714708800000,
#         "open": 150.25,
#         "high": 151.80,
#         "low": 149.50,
#         "close": 151.45,
#         "volume": 125000,
#     },
#     {
#         "Timestamp": 1714723200000,
#         "open": 151.45,
#         "high": 155.10,
#         "low": 151.20,
#         "close": 154.60,
#         "volume": 210000,
#     },
# ]


# df = pd.DataFrame(data)
# df = df.rename(columns={"Timestamp": "timestamp"})
# df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
# df.set_index("timestamp", inplace=True)

# new_df = pd.DataFrame(newData)
# new_df = new_df.rename(columns={"Timestamp": "timestamp"})
# new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], utc=True)
# new_df.set_index("timestamp", inplace=True)
# print(new_df)
# print("*" * 60)

# exchange = "kraken"
# symbol = "BTC/USDT"
# timeframe = "4h"
# root_dir = "data/raw"

# Test 1
# save_ohlcv(df, exchange, symbol, timeframe, root_dir)
# loaded = load_ohlcv(exchange, symbol, timeframe, root_dir)
# print(loaded is not None)
# print(validate_ohlcv(loaded,timeframe) is not None)
# print(loaded.index.tz)
# print(loaded.equals(df.sort_index()))

# Test 2
# loaded = load_ohlcv(exchange, symbol, timeframe, root_dir)
# print(loaded)

# Test 3
# merge = merge_ohlcv(df, new_df, "4h")
# print(merge)

# Test 4
# loaded = load_ohlcv(exchange, symbol, timeframe, root_dir)
# print(loaded is None)
# print(validate_ohlcv(new_df,timeframe) is not None)
# merged1 = upsert_ohlcv(new_df, exchange, symbol, timeframe, root_dir)
# print(merged1 is not None)
# print(merged1.index.has_duplicates == False)
# print(len(merged1) == len(new_df))
# print(load_ohlcv(exchange, symbol, timeframe, root_dir))

# print("*"*60)

# merged2 = upsert_ohlcv(new_df, exchange, symbol, timeframe, root_dir)
# print("Length not grow: ",len(merged2) == len(merged1) )
# print(merged2.index.has_duplicates == False)
# print(load_ohlcv(exchange, symbol, timeframe, root_dir))
