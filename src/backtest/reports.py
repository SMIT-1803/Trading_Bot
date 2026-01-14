from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import json


def _clean(obj):
    """Convert numpy scalars to python scalars for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_backtest_outputs(result: dict, out_dir: Path, k_stop: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames
    result["trades_df"].to_csv(out_dir / f"trades_k={str(k_stop)}.csv", index=False)
    result["equity_df"].to_csv(out_dir / f"equity_k={str(k_stop)}.csv")
    result["execution_df"].to_csv(out_dir / f"execution_df_k={str(k_stop)}.csv")

    for name in ["trade_metrics", "equity_metrics", "counts", "config"]:
        with (out_dir / f"{name}_k={str(k_stop)}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(_clean(result.get(name, {})), f, indent=2)


def build_report_pack(
    result: dict, out_dir: Path, *, k_stop: float, timeframe: str = "4h"
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    trades_df = result.get("trades_df")
    equity_df = result.get("equity_df")
    execution_df = result.get("execution_df")

    if equity_df is None or equity_df.empty:
        raise RuntimeError("REPORT ERROR: equity_df missing/empty")
    if execution_df is None or execution_df.empty:
        raise RuntimeError("REPORT ERROR: execution_df missing/empty")

    # Equity + Drawdown plots
    eq = equity_df.copy()
    eq.sort_index(inplace=True)
    eq = eq.dropna(subset=["equity"])
    equity_series = eq["equity"]

    # Equity plot
    plt.figure()
    plt.plot(equity_series.index, equity_series.values)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_dir / f"equity_curve_k={str(k_stop)}.png")
    plt.close()

    # Drawdown plot
    hwm = equity_series.cummax()
    dd = equity_series / hwm - 1.0

    plt.figure()
    plt.plot(dd.index, dd.values)
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (fraction)")
    plt.tight_layout()
    plt.savefig(out_dir / f"drawdown_k={str(k_stop)}.png")
    plt.close()

    # Trade Info

    if trades_df is None or trades_df.empty:
        with (out_dir / f"exit_reason_counts_k={str(k_stop)}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({}, f, indent=2)

        with (out_dir / f"hold_time_stats_k={str(k_stop)}.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump({"note": "No trades"}, f, indent=2)
        return

    t = trades_df.copy()
    if "reason" in t.columns:
        reason_counts = t["reason"].value_counts(dropna=False).to_dict()
    else:
        reason_counts = {}

    with (out_dir / f"exit_reason_counts_k={str(k_stop)}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(_clean(reason_counts), f, indent=2)

    if "reason" in t.columns and "pnl_usd" in t.columns:
        pnl_by_reason = (
            t.groupby("reason", dropna=False)
            .agg(
                num_trades=("pnl_usd", "size"),
                total_pnl=("pnl_usd", "sum"),
                avg_pnl=("pnl_usd", "mean"),
                median_pnl=("pnl_usd", "median"),
                mean_r=(
                    ("r_multiple", "mean")
                    if "r_multiple" in t.columns
                    else ("pnl_usd", "mean")
                ),
            )
            .sort_values("total_pnl", ascending=True)
        )
        pnl_by_reason.to_csv(out_dir / f"pnl_by_reason_k={str(k_stop)}.csv")
    else:
        pd.DataFrame().to_csv(
            out_dir / f"pnl_by_reason_k={str(k_stop)}.csv", index=False
        )

    # Hold time stats (bars)

    hold_stats = {}
    if "entry_time" in t.columns and "exit_time" in t.columns:
        entry = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
        exit_ = pd.to_datetime(t["exit_time"], utc=True, errors="coerce")
        hold_td = exit_ - entry

        tf = str(timeframe).lower().strip()
        if tf == "4h":
            bar_size = pd.Timedelta(hours=4)
        elif tf in ("1d", "1day", "d"):
            bar_size = pd.Timedelta(days=1)
        else:
            bar_size = pd.Timedelta(hours=4)

        bars_held = (hold_td / bar_size).astype(float)

        hold_stats["avg_bars_held"] = float(np.nanmean(bars_held))
        hold_stats["median_bars_held"] = float(np.nanmedian(bars_held))
        hold_stats["max_bars_held"] = float(np.nanmax(bars_held))

        if "pnl_usd" in t.columns:
            wins = t["pnl_usd"] > 0
            hold_stats["avg_bars_winners"] = (
                float(np.nanmean(bars_held[wins])) if wins.any() else np.nan
            )
            hold_stats["avg_bars_losers"] = (
                float(np.nanmean(bars_held[~wins])) if (~wins).any() else np.nan
            )
    with (out_dir / f"hold_time_stats_k={str(k_stop)}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(_clean(hold_stats), f, indent=2)

    # Histograms
    if "pnl_usd" in t.columns:
        plt.figure()
        plt.hist(t["pnl_usd"].dropna().values, bins=40)
        plt.title("PnL Histogram")
        plt.xlabel("PnL (USD)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"pnl_hist_k={str(k_stop)}.png")
        plt.close()

    if "r_multiple" in t.columns:
        plt.figure()
        plt.hist(t["r_multiple"].dropna().values, bins=40)
        plt.title("R-Multiple Histogram")
        plt.xlabel("R")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"r_multiple_hist_k={str(k_stop)}.png")
        plt.close()

    if "vol_regime" in execution_df.columns and "entry_time" in t.columns:
        ex = execution_df.sort_index()

        entry_ts = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
        regimes = ex["vol_regime"].reindex(entry_ts, method="ffill")
        t["entry_vol_regime"] = regimes.values

        if "pnl_usd" in t.columns:
            pnl_by_regime = (
                t.groupby("entry_vol_regime", dropna=False)
                .agg(
                    num_trades=("pnl_usd", "size"),
                    total_pnl=("pnl_usd", "sum"),
                    avg_pnl=("pnl_usd", "mean"),
                    mean_r=(
                        ("r_multiple", "mean")
                        if "r_multiple" in t.columns
                        else ("pnl_usd", "mean")
                    ),
                )
                .sort_values("total_pnl", ascending=True)
            )
            pnl_by_regime.to_csv(out_dir / f"pnl_by_vol_regime_k={str(k_stop)}.csv")
        else:
            pd.DataFrame().to_csv(
                out_dir / f"pnl_by_vol_regime_k={str(k_stop)}.csv", index=False
            )
    else:
        pd.DataFrame().to_csv(
            out_dir / f"pnl_by_vol_regime_k={str(k_stop)}.csv", index=False
        )
