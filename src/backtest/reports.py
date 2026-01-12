from pathlib import Path
import json

def save_backtest_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # DataFrames
    result["trades_df"].to_csv(out_dir / "trades.csv", index=False)
    result["equity_df"].to_csv(out_dir / "equity.csv")  # keeps timestamp index
    result["execution_df"].to_csv(out_dir / "execution_df.csv")  # big but simple

    # Dicts (convert numpy scalars -> float)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(x) for x in obj]
        # numpy scalar -> python scalar
        try:
            import numpy as np

            if isinstance(obj, np.generic):
                return obj.item()
        except Exception:
            pass
        return obj

    for name in ["trade_metrics", "equity_metrics", "counts", "config"]:
        with (out_dir / f"{name}.json").open("w", encoding="utf-8") as f:
            json.dump(clean(result.get(name, {})), f, indent=2)