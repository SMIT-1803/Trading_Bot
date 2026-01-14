import numpy as np
import pandas as pd
from pandas import DataFrame


def apply_buy_cost(price: float, cost_rate: float) -> float:
    filled_price = price * (1 + cost_rate)
    return filled_price


def apply_sell_cost(price: float, cost_rate: float) -> float:
    filled_price = price * (1 - cost_rate)
    return filled_price


def fill_entry_price(raw_entry_price: float, cost_rate: float) -> float:
    return apply_buy_cost(raw_entry_price, cost_rate)


def fill_exit_price(raw_exit_price: float, cost_rate: float) -> float:
    return apply_sell_cost(raw_exit_price, cost_rate)


def should_stop_out(low_price: float, stop_price: float) -> bool:
    if np.isfinite(low_price) and np.isfinite(stop_price):
        return low_price <= stop_price
    else:
        return False


def get_stop_exit_price(stop_price: float, cost_rate: float) -> float:
    return fill_exit_price(stop_price, cost_rate)


def get_close_exit_price(close_price: float, cost_rate: float) -> float:
    return fill_exit_price(close_price, cost_rate)


def run_simulation(
    df: DataFrame,
    initial_cash: float = 10000.0,
    risk_pct: float = 0.01,
    cost_rate: float = 0.00085,
    max_hold_bars: int = 12,
) -> tuple[DataFrame, DataFrame]:
    """
    Long-only, single-position backtest simulator with risk-based sizing
    and spot-market constraints.

    Core assumptions:
      - Only one position can be open at a time
      - No leverage (spot trading)
      - Position size is constrained by available cash
      - All prices include transaction costs (fees + slippage proxy)

    ENTRY LOGIC:
      - Enter when `entry_long == True`
      - Entry price = `entry_price` (typically candle close)
      - Filled entry price = entry_price * (1 + cost_rate)
      - Risk budget = equity * risk_pct * risk_multiplier
      - Units sized by:
            min(
                risk_budget / risk_per_unit,
                available_cash / filled_entry_price
            )

    POSITION MANAGEMENT:
      - Initial stop is fixed at `stop_price`
      - Partial take-profit (TP1):
            • Triggered when high >= entry_price + 1R
            • Sells `tp1_fraction` (default 50%)
            • Stop is moved to breakeven
      - Remaining position is managed independently after TP1

    EXIT PRIORITY (evaluated each bar, in order):
      1) STOP (intrabar):
            - If low <= stop_price → exit at stop_price (with costs)
      2) PERMISSION EXIT:
            - If can_trade == False → exit at close (with costs)
      3) TIME EXIT:
            - If bars_held >= max_hold_bars → exit at close (with costs)

    PNL & METRICS:
      - PnL includes partial and final exits
      - R-multiple = total_trade_pnl / (initial_units * risk_per_unit)
      - Equity is marked-to-market each bar using remaining units

    RETURNS:
      trades_df:
        One row per completed trade containing:
          - entry_time, exit_time
          - entry_price, exit_price
          - initial_units, remaining_units
          - exit_reason
          - total_pnl (USD)
          - partial_taken (bool)
          - R-multiple
          - cash_after

      equity_df:
        Time-indexed equity curve reflecting mark-to-market equity
        across the entire backtest.
    """
    df_4h = df.copy()
    df_4h.sort_index(inplace=True)

    required_columns = {
        "high",
        "low",
        "close",
        "entry_long",
        "entry_price",
        "stop_price",
        "risk_per_unit",
        "can_trade",
        "risk_multiplier",
        "ema_fast_4h",
        "vol_regime",
    }
    missing = required_columns - set(df_4h.columns)
    if missing:
        raise RuntimeError(f"INCORRECT DATAFRAME: missing columns {missing}")

    cash: float = float(initial_cash)
    equity: float = float(initial_cash)

    # position state
    in_position = False
    units = 0.0
    entry_time = None
    entry_price_filled = np.nan
    stop_price_state = np.nan
    risk_per_unit_state = np.nan
    bars_held = 0

    remaining_units = 0.0
    partial_taken = False
    tp1_price = np.nan
    tp1_fraction = 0.5
    realized_pnl_usd = 0.0
    regime = ""

    trades: list[dict] = []
    equity_records: list[dict] = []

    for ts, row in df_4h.iterrows():
        close = float(row["close"]) if np.isfinite(row["close"]) else np.nan
        low = float(row["low"]) if np.isfinite(row["low"]) else np.nan
        high = float(row["high"]) if np.isfinite(row["high"]) else np.nan
        if not in_position:
            # Flat: equity equals cash (no open position)
            equity = cash

            if bool(row["entry_long"]):
                regime = row["vol_regime"]
                partial_taken = False
                realized_pnl_usd = 0.0
                raw_entry = (
                    float(row["entry_price"])
                    if np.isfinite(row["entry_price"])
                    else np.nan
                )
                rpu = (
                    float(row["risk_per_unit"])
                    if np.isfinite(row["risk_per_unit"])
                    else np.nan
                )
                rm = (
                    float(row["risk_multiplier"])
                    if np.isfinite(row["risk_multiplier"])
                    else 0.0
                )

                # Basic validity checks
                if (
                    (not np.isfinite(raw_entry))
                    or (not np.isfinite(rpu))
                    or (rpu <= 0)
                    or (rm <= 0)
                ):
                    equity_records.append({"timestamp": ts, "equity": equity})
                    continue

                filled_entry = apply_buy_cost(raw_entry, cost_rate)

                if (not np.isfinite(filled_entry)) or (filled_entry <= 0):
                    equity_records.append({"timestamp": ts, "equity": equity})
                    continue

                # Risk budget (equity-based) adjusted by volatility regime multiplier
                risk_budget = equity * float(risk_pct)
                adj_risk_budget = risk_budget * rm

                # Units sized by risk, capped by available cash (spot-only)
                units_by_risk = adj_risk_budget / rpu
                units_by_cash = cash / filled_entry
                final_units = min(units_by_risk, units_by_cash)

                # Skip if cannot buy anything meaningful
                if (not np.isfinite(final_units)) or (final_units <= 0):
                    equity_records.append({"timestamp": ts, "equity": equity})
                    continue

                # Apply entry (cash decreases by notional)
                cash -= final_units * filled_entry

                # Set position state
                in_position = True

                units = float(final_units)
                remaining_units = units
                entry_time = ts
                entry_price_filled = float(filled_entry)
                stop_price_state = (
                    float(row["stop_price"])
                    if np.isfinite(row["stop_price"])
                    else np.nan
                )
                risk_per_unit_state = float(rpu)
                tp1_price = (
                    float(raw_entry) + 1.0 * risk_per_unit_state
                    if np.isfinite(raw_entry) and np.isfinite(risk_per_unit_state)
                    else np.nan
                )
                bars_held = 0

        else:
            # In position: default mark-to-market equity
            if np.isfinite(close):
                equity = cash + remaining_units * close
            else:
                equity = (
                    cash  # fallback if close is bad (shouldn't happen after validation)
                )

            ema_now = (
                float(row["ema_fast_4h"]) if np.isfinite(row["ema_fast_4h"]) else np.nan
            )

            if partial_taken and np.isfinite(ema_now):
                stop_price_state = (
                    ema_now
                    if not np.isfinite(stop_price_state)
                    else max(stop_price_state, ema_now)
                )

            # Exit checks (priority order)
            exit_trade = False
            exit_reason = None
            raw_exit = np.nan

            # 1) STOP (intrabar)
            if (
                np.isfinite(low)
                and np.isfinite(stop_price_state)
                and (low <= stop_price_state)
            ):
                raw_exit = stop_price_state
                exit_reason = "STOP"
                exit_trade = True

            # 2) TP Price Hit
            elif (
                np.isfinite(high)
                and np.isfinite(tp1_price)
                and high >= tp1_price
                and partial_taken == False
            ):
                raw_tp1_exit = tp1_price
                filled_tp1_exit = apply_sell_cost(raw_tp1_exit, cost_rate)
                units_sold = remaining_units * tp1_fraction
                remaining_units -= units_sold
                cash += units_sold * filled_tp1_exit
                if np.isfinite(ema_now):
                    stop_price_state = (
                        ema_now
                        if not np.isfinite(stop_price_state)
                        else max(stop_price_state, ema_now)
                    )
                realized_pnl_usd += units_sold * (filled_tp1_exit - entry_price_filled)

                partial_taken = True
                if np.isfinite(filled_tp1_exit):
                    equity = cash + remaining_units * filled_tp1_exit

            # 3) PERMISSION flip (close)
            elif (not bool(row["can_trade"])) and np.isfinite(close):
                raw_exit = close
                exit_reason = "PERMISSION"
                exit_trade = True

            # 4) TIME stop (close)
            elif (
                bars_held >= (int(max_hold_bars) if regime == "HIGH_VOL" else 10)
            ) and np.isfinite(close):
                raw_exit = close
                exit_reason = "TIME"
                exit_trade = True

            if exit_trade:
                filled_exit = apply_sell_cost(float(raw_exit), cost_rate)

                final_pnl_usd = remaining_units * (filled_exit - entry_price_filled)
                total_pnl = realized_pnl_usd + final_pnl_usd
                cash += remaining_units * filled_exit

                # R multiple: PnL divided by initial risk (units * risk_per_unit)
                denom = units * risk_per_unit_state
                r_multiple = (
                    total_pnl / denom if (np.isfinite(denom) and denom > 0) else np.nan
                )

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "entry_price": entry_price_filled,
                        "exit_price": float(filled_exit),
                        "units": units,
                        "remaining_units": remaining_units,
                        "reason": exit_reason,
                        "pnl_usd": float(total_pnl),
                        "partial_taken": partial_taken,
                        "r_multiple": (
                            float(r_multiple) if np.isfinite(r_multiple) else np.nan
                        ),
                        "cash_after": float(cash),
                    }
                )

                # Reset position state
                in_position = False
                units = 0.0
                entry_time = None
                entry_price_filled = np.nan
                stop_price_state = np.nan
                risk_per_unit_state = np.nan
                bars_held = 0

                partial_taken = False
                realized_pnl_usd = 0.0
                tp1_price = np.nan
                remaining_units = 0.0

                equity = cash  # now flat again

            else:
                # Still holding
                bars_held += 1

        equity_records.append({"timestamp": ts, "equity": float(equity)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records).set_index("timestamp")
    return trades_df, equity_df
