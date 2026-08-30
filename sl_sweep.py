"""Compare stop-loss methods for the Balanced Elite Hybrid Strategy.

Entry logic, TP (22 points), trailing lock and the 2-position grid are held
constant; only the stop-loss method varies.

Usage:
    python sl_sweep.py data.csv [--lot 0.01] [--spread-points 20]
"""

import argparse

import pandas as pd

import backtest as bt


def variants():
    out = [("no stop (TP/trail only)", dict(sl_mode="none"))]
    for cash in (5, 10, 20, 40):
        out.append((f"fixed cash -${cash}", dict(sl_mode="cash", sl_cash=float(cash))))
    for pts in (10, 22, 44, 66, 100, 150, 250, 400):
        out.append((f"price SL {pts} pt", dict(sl_mode="points", sl_points=float(pts))))
    for mult in (0.5, 1.0, 2.0, 3.0, 5.0):
        out.append((f"ATR(14) x {mult}", dict(sl_mode="atr", atr_mult=mult)))
    for bars_held in (15, 60, 240):
        out.append((f"time stop {bars_held} bars, no SL",
                    dict(sl_mode="none", max_hold_bars=bars_held)))
    for pts, bars_held in ((66, 60), (100, 240)):
        out.append((f"price SL {pts} pt + time stop {bars_held}",
                    dict(sl_mode="points", sl_points=float(pts), max_hold_bars=bars_held)))
    return out


def evaluate(bars, balance, lot, point, contract_size, spread_points, commission, cfg):
    trades, equity, max_dd, _, stats = bt.run(
        bars, balance, point, contract_size, spread_points, commission, lot,
        sl_mode=cfg.get("sl_mode", "cash"),
        sl_cash=cfg.get("sl_cash", bt.HARD_STOP_CASH),
        sl_points=cfg.get("sl_points", 22.0),
        atr_mult=cfg.get("atr_mult", 1.0),
        atr_period=cfg.get("atr_period", 14),
        max_hold_bars=cfg.get("max_hold_bars", 0),
    )
    df = pd.DataFrame(trades)
    row = {
        "net": equity - balance,
        "net_incl_float": stats["float_equity"] - balance,
        "max_dd": max_dd,
        "max_float_dd": stats["max_float_dd"],
        "min_float_equity": stats["min_float_equity"],
        "margin_call": "" if stats["margin_call_at"] is None
        else str(pd.to_datetime(stats["margin_call_at"]).date()),
        "trades": len(df),
        "win_rate": float("nan"),
        "pf": float("nan"),
        "avg": float("nan"),
        "worst": float("nan"),
        "blown": "",
    }
    if not df.empty:
        wins = df[df["pnl"] > 0]["pnl"]
        losses = -df[df["pnl"] <= 0]["pnl"]
        row["win_rate"] = len(wins) / len(df) * 100
        row["pf"] = wins.sum() / losses.sum() if losses.sum() > 0 else float("inf")
        row["avg"] = df["pnl"].mean()
        row["worst"] = df["pnl"].min()
        # equity path with no further trading once the deposit is gone
        blown = df[df["equity"] <= 0]
        if not blown.empty:
            row["blown"] = str(pd.to_datetime(blown["closed"].iloc[0]).date())
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--balance", type=float, default=200.0)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--contract-size", type=float, default=100.0)
    ap.add_argument("--spread-points", type=float, default=20.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--out", default="sl_sweep.csv")
    args = ap.parse_args()

    bars = bt.load_bars(args.csv)
    rows = []
    for name, cfg in variants():
        row = evaluate(bars, args.balance, args.lot, args.point, args.contract_size,
                       args.spread_points, args.commission_per_lot, cfg)
        row["method"] = name
        rows.append(row)
        print(f"{name:<34} net {row['net']:>10.2f}  incl.float {row['net_incl_float']:>10.2f}  "
              f"floatDD {row['max_float_dd']:>9.2f}  n {row['trades']:>5}  "
              f"win {row['win_rate']:>6.2f}%  pf {row['pf']:>5.2f}  "
              f"worst {row['worst']:>8.2f}  margin call {row['margin_call']}", flush=True)

    df = pd.DataFrame(rows)[
        ["method", "net", "net_incl_float", "max_dd", "max_float_dd",
         "min_float_equity", "margin_call", "trades", "win_rate", "pf", "avg", "worst"]
    ].sort_values("net_incl_float", ascending=False)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
