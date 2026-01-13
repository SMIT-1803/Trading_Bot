import numpy as np
import pandas as pd
from pandas import DataFrame


def compute_equity_metrics(equity_df: DataFrame) -> dict:
    df = equity_df.copy()
    df.sort_index(inplace=True)
    df = df.dropna(subset=["equity"])
    if len(df) >= 2:
        eq = df["equity"]

        initial_equity = eq.iloc[0]
        final_equity = eq.iloc[-1]

        if initial_equity > 0:
            total_return_pct = final_equity / initial_equity - 1
        else:
            raise RuntimeError("INITIAL EQUITY SHOULD BE POSITIVE")

        hwm = eq.cummax()
        dd = eq / hwm - 1

        max_drawdown_pct = dd.min()

        underwater = eq < hwm

        duration = 0
        max_drawdown_duration_bars = duration
        for val in underwater:
            if not val:
                max_drawdown_duration_bars = max(max_drawdown_duration_bars, duration)
                duration = 0
            else:
                duration += 1
        max_drawdown_duration_bars = max(max_drawdown_duration_bars, duration)
        ret = eq.pct_change()
        mean_return = ret.mean()
        std_return = ret.std()

        return {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration_bars": max_drawdown_duration_bars,
            "mean_bar_return": mean_return,
            "std_bar_return": std_return,
            "number_of_bars": len(eq),
        }
    else:
        raise RuntimeError("EQUITY ERROR: need at least 2 non-NaN equity points")


def compute_trade_metrics(trades_df: DataFrame) -> dict:
    if trades_df is None or len(trades_df) == 0:
        raise RuntimeError("TRADE METRICS ERROR: no trades to analyze")

    df = trades_df.copy()
    num_trades = len(df)

    wins = df["pnl_usd"] > 0
    losses = df["pnl_usd"] < 0

    num_wins = int(wins.sum())
    num_losses = int(losses.sum())
    win_rate = num_wins / num_trades

    r = df["r_multiple"]
    expectancy_r = r.mean()

    profits_series = df.loc[wins, "pnl_usd"]
    losses_series = df.loc[losses, "pnl_usd"]

    profits = float(profits_series.sum()) if len(profits_series) else 0.0
    abs_losses = float((-losses_series).sum()) if len(losses_series) else 0.0

    if abs_losses == 0.0:
        profit_factor = np.inf if profits > 0 else np.nan
    else:
        profit_factor = profits / abs_losses

    avg_win = float(profits_series.mean()) if len(profits_series) else np.nan
    avg_loss = float(losses_series.mean()) if len(losses_series) else np.nan
    avg_win_loss = (avg_win / abs(avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else np.nan

    # bars held (4H bars)
    bar_size = pd.Timedelta(hours=4)
    hold_td = df["exit_time"] - df["entry_time"]
    bars_held = (hold_td / bar_size)

    avg_bars_held = float(bars_held.mean())
    max_bars_held = float(bars_held.max())

    return {
        "num_trades": num_trades,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "win_rate": win_rate,

        "expectancy_r": expectancy_r,
        "median_r_multiple": r.median(),
        "max_r_multiple": r.max(),
        "min_r_multiple": r.min(),
        "std_r_multiple": r.std(),

        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss": avg_win_loss,

        "avg_bars_held": avg_bars_held,
        "max_bars_held": max_bars_held,
    }