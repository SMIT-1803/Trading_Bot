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
