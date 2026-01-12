from src.data_layer.storage import load_ohlcv
from src.orchestration.pipeline import build_execution_frame
from src.backtest.simulator import run_simulation
from src.backtest.metrics import compute_trade_metrics, compute_equity_metrics
from src.backtest.reports import save_backtest_outputs

from pathlib import Path
import yaml
from datetime import datetime, timezone


def load_settings(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"SETTINGS ERROR: settings file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise RuntimeError("SETTINGS ERROR: YAML must parse into a dict.")
    return cfg


def settings_to_job_args(cfg: dict) -> dict:
    exchange = cfg.get("exchange")
    symbol = cfg.get("symbol")
    if not exchange or not symbol:
        raise RuntimeError("SETTINGS ERROR: 'exchange' and 'symbol' are required.")

    data = cfg.get("data", {})
    features = cfg.get("features", {})
    strategy = cfg.get("strategy", {})
    backtest = cfg.get("backtest", {})

    root_dir = data.get("root_dir")
    if not root_dir:
        raise RuntimeError("SETTINGS ERROR: data.root_dir is required.")

    timeframe_4h = data.get("timeframe_4h", "4h")
    timeframe_1d = data.get("timeframe_1d", "1d")

    ema_span_1d = int(features.get("ema_span_1d", 200))
    atr_window = int(features.get("atr_window", 14))

    # lookback_days in your YAML is for the volatility percentile lookback on 4H bars
    lookback_days = int(features.get("lookback_days", 60))
    pct_window = lookback_days * 6  # 4H => 6 bars/day

    # YAML uses 35/65 (percent). Convert to 0.35/0.65
    low_vol_pct = float(features.get("low_vol_pct", 35)) / 100.0
    high_vol_pct = float(features.get("high_vol_pct", 65)) / 100.0
    thresholds = (low_vol_pct, high_vol_pct)

    ema_fast = int(strategy.get("ema_fast", 50))
    ema_slow = int(strategy.get("ema_slow", 200))
    k_stop = float(strategy.get("k_stop", 1.5))

    initial_cash = float(backtest.get("initial_cash", 10000.0))
    risk_pct = float(backtest.get("risk_pct", 0.01))
    cost_rate = float(backtest.get("cost_rate", 0.00085))
    max_hold_bars = int(backtest.get("max_hold_bars", 12))

    return {
        "exchange": exchange,
        "symbol": symbol,
        "root_dir": root_dir,
        "timeframe_4h": timeframe_4h,
        "timeframe_1d": timeframe_1d,
        "ema_span_1d": ema_span_1d,
        "atr_window": atr_window,
        "pct_window": pct_window,
        "thresholds": thresholds,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "k_stop": k_stop,
        "initial_cash": initial_cash,
        "risk_pct": risk_pct,
        "cost_rate": cost_rate,
        "max_hold_bars": max_hold_bars,
    }


def run_backtest_job(
    exchange: str,
    symbol: str,
    root_dir: str,
    *,
    timeframe_4h: str,
    timeframe_1d: str,
    ema_span_1d: int,
    atr_window: int,
    pct_window: int,
    thresholds: tuple[float, float],
    ema_fast: int,
    ema_slow: int,
    k_stop: float,
    initial_cash: float,
    risk_pct: float,
    cost_rate: float,
    max_hold_bars: int,
) -> dict:
    df_4h = load_ohlcv(
        exchange=exchange, symbol=symbol, timeframe=timeframe_4h, root_dir=root_dir
    )
    df_1d = load_ohlcv(
        exchange=exchange, symbol=symbol, timeframe=timeframe_1d, root_dir=root_dir
    )

    if df_4h is None or df_4h.empty:
        raise RuntimeError("BACKTEST ERROR: 4H dataframe is missing/empty.")
    if df_1d is None or df_1d.empty:
        raise RuntimeError("BACKTEST ERROR: 1D dataframe is missing/empty.")

    df_4h = df_4h.sort_index()
    df_1d = df_1d.sort_index()

    execution_df = build_execution_frame(
        df_4h,
        df_1d,
        ema_span_1d=ema_span_1d,
        atr_window=atr_window,
        pct_window=pct_window,
        thresholds=thresholds,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        k_stop=k_stop,
    )

    entry_long_count = int(execution_df["entry_long"].sum())
    can_trade_count = int(execution_df["can_trade"].sum())

    trades_df, equity_df = run_simulation(
        df=execution_df,
        initial_cash=initial_cash,
        risk_pct=risk_pct,
        cost_rate=cost_rate,
        max_hold_bars=max_hold_bars,
    )

    equity_metrics = compute_equity_metrics(equity_df)

    if trades_df is None or trades_df.empty:
        trade_metrics = {}
        num_trades = 0
    else:
        trade_metrics = compute_trade_metrics(trades_df)
        num_trades = int(len(trades_df))

    return {
        "execution_df": execution_df,
        "trades_df": trades_df,
        "equity_df": equity_df,
        "trade_metrics": trade_metrics,
        "equity_metrics": equity_metrics,
        "counts": {
            "entry_long": entry_long_count,
            "can_trade": can_trade_count,
            "trades": num_trades,
        },
        "config": {
            "exchange": exchange,
            "symbol": symbol,
            "root_dir": root_dir,
            "timeframe_4h": timeframe_4h,
            "timeframe_1d": timeframe_1d,
            "ema_span_1d": ema_span_1d,
            "atr_window": atr_window,
            "pct_window": pct_window,
            "thresholds": thresholds,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "k_stop": k_stop,
            "initial_cash": initial_cash,
            "risk_pct": risk_pct,
            "cost_rate": cost_rate,
            "max_hold_bars": max_hold_bars,
        },
    }

def main():
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_settings(project_root / "config" / "setting.yaml")
    args = settings_to_job_args(cfg)

    result = run_backtest_job(**args)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_symbol = args["symbol"].replace("/", "-")
    out_dir = (
        project_root
        / "data"
        / "cache"
        / "backtests"
        / f"{args['exchange']}_{safe_symbol}_{args['timeframe_4h']}_{run_id}"
    )
    save_backtest_outputs(result, out_dir)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
