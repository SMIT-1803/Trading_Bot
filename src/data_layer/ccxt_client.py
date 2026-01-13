import ccxt
from datetime import datetime, timezone
import pandas as pd
from pandas import DataFrame
from validators import validate_ohlcv
from storage import load_ohlcv, upsert_ohlcv


def get_exchange_client(exchange: str) -> ccxt.Exchange:
    if not hasattr(ccxt, exchange):
        raise ValueError(f"Unsupported exchange '{exchange}' in ccxt.")
    client_class = getattr(ccxt, exchange)
    client = client_class(
        {
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        }
    )

    client.load_markets()
    return client


def fetch_ohlcv_chunk(
    client: ccxt.Exchange, symbol: str, timeframe: str, start_ms: int, limit: int
) -> DataFrame:
    ohlcv = client.fetch_ohlcv(symbol, timeframe, start_ms, limit)

    if not ohlcv:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    try:
        df = validate_ohlcv(df, timeframe)
    except RuntimeError as e:
        raise RuntimeError(f"DATA FETCH ERROR: {e}")

    return df


def fetch_ohlcv_range(
    client, symbol: str, timeframe: str, start_ms: int, end_ms: int, limit: int
) -> DataFrame:
    since = start_ms
    chunks: list[DataFrame] = []

    while since < end_ms:
        chunk = fetch_ohlcv_chunk(client, symbol, timeframe, since, limit)
        if chunk.empty:
            break

        end_ts = pd.to_datetime(end_ms, unit="ms", utc=True)
        chunk = chunk.loc[chunk.index <= end_ts]
        if chunk.empty:
            break

        chunks.append(chunk)
        last_ms = int(chunk.index[-1].timestamp() * 1000)
        since = last_ms + 1

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    df = validate_ohlcv(df, timeframe)
    return df


def update_ohlcv(
    exchange: str, symbol: str, timeframe: str, root_dir: str
) -> DataFrame:
    client = get_exchange_client(exchange)
    old_df = load_ohlcv(exchange, symbol, timeframe, root_dir)

    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)

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

    tf_delta = tf_map[tf]

    if old_df is None:
        start_ts = now - pd.Timedelta(days=730)  # Approx 2 years backfill
        start_ms = int(start_ts.timestamp() * 1000)
        new_df = fetch_ohlcv_range(client, symbol, tf, start_ms, now_ms, limit=500)

        if new_df.empty:
            raise RuntimeError(
                f"UPDATE ERROR: No OHLCV fetched for {exchange} {symbol} {tf}. Check symbol/timeframe."
            )
    else:
        last_ts = old_df.index[-1]
        overlap_candles = 10
        start_ts = last_ts - overlap_candles * tf_delta
        start_ms = int(start_ts.timestamp() * 1000)

        new_df = fetch_ohlcv_range(client, symbol, tf, start_ms, now_ms, limit=500)
        if new_df.empty:
            return old_df

    now_ts = pd.Timestamp(now)
    new_df = new_df.loc[new_df.index + tf_delta <= now_ts]
    updated = upsert_ohlcv(new_df, exchange, symbol, tf, root_dir)
    return updated


############################## TESTS ##############################

# df = update_ohlcv("kraken", "BTC/USD", "4h", "data/raw")
# print(df)
# print(df.index.tz)
# print(df.index.is_monotonic_increasing)
# print(df.index.has_duplicates)
