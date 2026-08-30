"""Sweep take-profit method, grid size and grid step distance.

Stage 1 varies the TP method with the grid held at the spec values.
Stage 2 varies max positions x grid step at the best TP from stage 1.
Stage 3 re-runs the top stage-2 combinations across spreads as a robustness check.

Entry logic is unchanged; the stop is held at the best performer from the SL
sweep (ATR(14) x 3) unless --sl-mode says otherwise.

Usage:
    python tp_grid_sweep.py data.csv [--lot 0.01] [--spread-points 20] [--jobs 8]
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

import backtest as bt

BARS = None  # populated per worker process


def init_worker(csv_path):
    global BARS
    BARS = bt.load_bars(csv_path)


def evaluate(job):
    label, cfg, ctx = job
    trades, equity, max_dd, _, stats = bt.run(
        BARS, ctx["balance"], ctx["point"], ctx["contract_size"],
        ctx["spread_points"], ctx["commission"], ctx["lot"],
        sl_mode=cfg.get("sl_mode", ctx["sl_mode"]),
        sl_cash=cfg.get("sl_cash", ctx["sl_cash"]),
        sl_points=cfg.get("sl_points", 22.0),
        atr_mult=cfg.get("atr_mult", ctx["atr_mult"]),
        atr_period=14,
        max_hold_bars=cfg.get("max_hold_bars", 0),
        tp_mode=cfg.get("tp_mode", "points"),
        tp_points=cfg.get("tp_points", bt.TP_POINTS),
        tp_atr_mult=cfg.get("tp_atr_mult", 1.0),
        tp_rr=cfg.get("tp_rr", 1.0),
        max_positions=cfg.get("max_positions", bt.MAX_POSITIONS),
        grid_step_points=cfg.get("grid_step_points", bt.GRID_STEP_POINTS),
        trail_activate=cfg.get("trail_activate", bt.TRAIL_ACTIVATE),
        trail_retrace=cfg.get("trail_retrace", bt.TRAIL_RETRACE),
    )
    df = pd.DataFrame(trades)
    row = {
        "config": label,
        "net": equity - ctx["balance"],
        "net_incl_float": stats["float_equity"] - ctx["balance"],
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
    }
    row.update({f"p_{k}": v for k, v in cfg.items()})
    if not df.empty:
        wins = df[df["pnl"] > 0]["pnl"]
        losses = -df[df["pnl"] <= 0]["pnl"]
        row["win_rate"] = len(wins) / len(df) * 100
        row["pf"] = wins.sum() / losses.sum() if losses.sum() > 0 else float("inf")
        row["avg"] = df["pnl"].mean()
        row["worst"] = df["pnl"].min()
    return row


def tp_variants():
    out = []
    for pts in (22, 50, 100, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1500):
        out.append((f"TP {pts} pt", dict(tp_mode="points", tp_points=float(pts))))
    for mult in (0.5, 1.0, 2.0, 4.0, 8.0):
        out.append((f"TP ATR(14) x {mult}", dict(tp_mode="atr", tp_atr_mult=mult)))
    for rr in (0.5, 1.0, 1.5, 2.0, 3.0):
        out.append((f"TP {rr} x SL (R:R)", dict(tp_mode="rr", tp_rr=rr)))
    out.append(("TP none (trail lock only)", dict(tp_mode="none")))
    out.append(("TP none, trail 3/1",
                dict(tp_mode="none", trail_activate=3.0, trail_retrace=1.0)))
    return out


def grid_variants(best_tp_cfg):
    out = []
    for positions, step in itertools.product((1, 2, 3, 5), (12, 30, 60, 120, 250, 500)):
        if positions == 1 and step != 12:
            continue  # step is irrelevant with a single position
        cfg = dict(best_tp_cfg)
        cfg.update(max_positions=positions, grid_step_points=float(step))
        label = f"{positions} pos, step {step} pt" if positions > 1 else "1 pos (no grid)"
        out.append((label, cfg))
    return out


def run_stage(name, jobs, ctx, executor, out_path):
    rows = list(executor.map(evaluate, [(lbl, cfg, ctx) for lbl, cfg in jobs]))
    df = pd.DataFrame(rows).sort_values("net_incl_float", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"\n=== {name} ===")
    cols = ["config", "net", "net_incl_float", "max_float_dd", "margin_call",
            "trades", "win_rate", "pf", "worst"]
    print(df[cols].to_string(index=False))
    print(f"-> {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--balance", type=float, default=200.0)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--contract-size", type=float, default=100.0)
    ap.add_argument("--spread-points", type=float, default=20.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--sl-mode", default="atr")
    ap.add_argument("--sl-cash", type=float, default=10.0)
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--prefix", default="tp_grid")
    args = ap.parse_args()

    ctx = {
        "balance": args.balance, "lot": args.lot, "point": args.point,
        "contract_size": args.contract_size, "spread_points": args.spread_points,
        "commission": args.commission_per_lot, "sl_mode": args.sl_mode,
        "sl_cash": args.sl_cash, "atr_mult": args.atr_mult,
    }

    init_worker(args.csv)  # stage 3 runs in this process

    with ProcessPoolExecutor(args.jobs, initializer=init_worker,
                             initargs=(args.csv,)) as ex:
        tp_df = run_stage("Stage 1: TP method", tp_variants(), ctx, ex,
                          f"{args.prefix}_stage1_tp.csv")

        best_label = tp_df.iloc[0]["config"]
        best_cfg = dict(next(c for l, c in tp_variants() if l == best_label))
        print(f"\nbest TP -> {best_label}: {best_cfg}")

        grid_df = run_stage(f"Stage 2: grid at {best_label}",
                            grid_variants(best_cfg), ctx, ex,
                            f"{args.prefix}_stage2_grid.csv")

        top = grid_df.head(3)
        jobs = []
        for _, row in top.iterrows():
            cfg = dict(best_cfg)
            cfg.update(max_positions=int(row["p_max_positions"]),
                       grid_step_points=float(row["p_grid_step_points"]))
            for spread in (0, 10, 20, 30):
                c = dict(cfg)
                jobs.append((f"{row['config']} @ spread {spread}", c))
        # stage 3 needs a per-job spread, so run it serially per spread setting
        rows = []
        for label, cfg in jobs:
            spread = float(label.rsplit(" ", 1)[1])
            ctx3 = dict(ctx, spread_points=spread)
            rows.append(evaluate((label, cfg, ctx3)))
        df3 = pd.DataFrame(rows).sort_values("net_incl_float", ascending=False)
        df3.to_csv(f"{args.prefix}_stage3_spread.csv", index=False)
        print("\n=== Stage 3: top combos across spreads ===")
        print(df3[["config", "net", "net_incl_float", "max_float_dd", "margin_call",
                   "trades", "win_rate", "pf"]].to_string(index=False))


if __name__ == "__main__":
    main()
