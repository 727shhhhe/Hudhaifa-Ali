"""Bar-by-bar backtest of the Balanced Elite Hybrid Strategy (XAUUSD M1).

Mirrors BalancedEliteHybrid.mq5: signals are evaluated on closed bars,
basket management (hard stop -> micro-trail -> TP -> grid add) is checked
intrabar in OHLC order.

Usage:
    python backtest.py data.csv [--balance 200] [--point 0.01] [--contract-size 100]

CSV must contain datetime/open/high/low/close columns (header names are
matched case-insensitively; a single combined datetime column or separate
date+time columns both work).
"""

import argparse
import sys

import pandas as pd

MOM_PERIOD = 9
BODY_EXPAND_MIN = 1.50
BODY_EXPAND_MAX = 4.00
BASE_LOT = 0.01
MAX_POSITIONS = 2
GRID_STEP_POINTS = 12.0
TP_POINTS = 22.0
TRAIL_ACTIVATE = 12.00
TRAIL_RETRACE = 4.00
HARD_STOP_CASH = 10.00

BUY, SELL = 1, -1


def load_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().lstrip("<").rstrip(">") for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        stamp = df["date"].astype(str) + " " + df["time"].astype(str)
    else:
        col = next((c for c in ("datetime", "date", "timestamp", "time") if c in df.columns), None)
        if col is None:
            sys.exit(f"no datetime column found in {list(df.columns)}")
        stamp = df[col].astype(str)

    out = pd.DataFrame({"datetime": pd.to_datetime(stamp, format="mixed", dayfirst=False)})
    for name in ("open", "high", "low", "close"):
        if name not in df.columns:
            sys.exit(f"missing '{name}' column in {list(df.columns)}")
        out[name] = pd.to_numeric(df[name], errors="coerce")

    out = out.dropna().sort_values("datetime").reset_index(drop=True)
    if out.empty:
        sys.exit("no usable bars after parsing")
    return out


class Basket:
    def __init__(self, direction: int, price: float, volume: float):
        self.direction = direction
        self.entries = [(price, volume)]
        self.peak_profit = 0.0

    def add(self, price: float, volume: float) -> None:
        self.entries.append((price, volume))

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def volume(self) -> float:
        return sum(v for _, v in self.entries)

    @property
    def avg_price(self) -> float:
        return sum(p * v for p, v in self.entries) / self.volume

    @property
    def last_price(self) -> float:
        return self.entries[-1][0]

    def profit(self, price: float, contract_size: float) -> float:
        return sum(
            (price - p) * self.direction * v * contract_size for p, v in self.entries
        )


def run(bars: pd.DataFrame, balance: float, point: float, contract_size: float,
        spread_points: float, commission_per_lot: float, lot: float = BASE_LOT):
    spread = spread_points * point
    equity = balance
    basket = None
    trades = []
    equity_curve = []
    peak_equity = balance
    max_dd = 0.0

    def close_basket(price: float, when, reason: str) -> None:
        nonlocal basket, equity, peak_equity, max_dd
        pnl = basket.profit(price, contract_size)
        pnl -= commission_per_lot * basket.volume
        equity += pnl
        trades.append(
            {
                "closed": when,
                "direction": "buy" if basket.direction == BUY else "sell",
                "positions": basket.count,
                "avg_price": basket.avg_price,
                "exit_price": price,
                "pnl": pnl,
                "reason": reason,
                "equity": equity,
            }
        )
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, peak_equity - equity)
        basket = None

    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    t = bars["datetime"].to_numpy()

    for i in range(MOM_PERIOD + 2, len(bars)):
        # --- manage an open basket across the bar (open, high, low, close)
        if basket is not None:
            # adverse extreme first, so a stop and a target inside the same bar
            # resolve pessimistically
            adverse = l[i] if basket.direction == BUY else h[i]
            favorable = h[i] if basket.direction == BUY else l[i]

            for step_index, price in enumerate((o[i], adverse, favorable, c[i])):
                if basket is None:
                    break
                is_open = step_index == 0
                long_side = basket.direction == BUY
                mark = price if long_side else price + spread

                def fill(threshold: float) -> float:
                    """Fill at the threshold, or at the gapped bar open if worse."""
                    if not is_open:
                        return threshold
                    return min(threshold, mark) if long_side else max(threshold, mark)

                floating = basket.profit(mark, contract_size)
                vol_value = basket.volume * contract_size
                avg = basket.avg_price

                sl_price = avg - HARD_STOP_CASH / vol_value if long_side \
                    else avg + HARD_STOP_CASH / vol_value
                if (mark <= sl_price) if long_side else (mark >= sl_price):
                    close_basket(fill(sl_price), t[i], "hard_stop")
                    continue

                if floating >= TRAIL_ACTIVATE:
                    basket.peak_profit = max(basket.peak_profit, floating)
                if (basket.peak_profit >= TRAIL_ACTIVATE
                        and basket.peak_profit - floating >= TRAIL_RETRACE):
                    close_basket(mark, t[i], "trail_lock")
                    continue

                tp_price = avg + TP_POINTS * point if long_side else avg - TP_POINTS * point
                if (mark >= tp_price) if long_side else (mark <= tp_price):
                    close_basket(fill(tp_price), t[i], "take_profit")
                    continue

                if basket.count < MAX_POSITIONS:
                    grid = GRID_STEP_POINTS * point
                    grid_price = basket.last_price - grid if long_side \
                        else basket.last_price + grid
                    if (mark <= grid_price) if long_side else (mark >= grid_price):
                        basket.add(fill(grid_price), lot)

        equity_curve.append((t[i], equity))

        if basket is not None:
            continue

        # --- signal on the just-closed bar i (prev = i-1), entry at bar i+1 open
        if i + 1 >= len(bars):
            break

        mom = c[i] - c[i - MOM_PERIOD]
        mom_slope = mom - (c[i - 1] - c[i - 1 - MOM_PERIOD])

        body = abs(c[i] - o[i])
        prev_body = abs(c[i - 1] - o[i - 1])
        if prev_body <= 0:
            continue
        ratio = body / prev_body
        expansion = BODY_EXPAND_MIN <= ratio <= BODY_EXPAND_MAX
        opposite = (c[i] > o[i]) != (c[i - 1] > o[i - 1])

        if not (expansion and opposite):
            continue

        entry_bid = o[i + 1]
        if mom > 0 and mom_slope >= 0:
            basket = Basket(BUY, entry_bid + spread, lot)
        elif mom < 0 and mom_slope <= 0:
            basket = Basket(SELL, entry_bid, lot)

    return trades, equity, max_dd, equity_curve


def report(trades, start_balance, equity, max_dd, bars):
    df = pd.DataFrame(trades)
    lines = []
    lines.append(f"bars:            {len(bars)}  ({bars['datetime'].iloc[0]} -> {bars['datetime'].iloc[-1]})")
    lines.append(f"start balance:   {start_balance:.2f}")
    lines.append(f"end equity:      {equity:.2f}")
    lines.append(f"net profit:      {equity - start_balance:.2f}")
    lines.append(f"max drawdown:    {max_dd:.2f}")
    lines.append(f"baskets closed:  {len(df)}")
    if not df.empty:
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        gross_win = wins["pnl"].sum()
        gross_loss = -losses["pnl"].sum()
        lines.append(f"win rate:        {len(wins) / len(df) * 100:.2f}%  ({len(wins)}W / {len(losses)}L)")
        lines.append(f"gross profit:    {gross_win:.2f}")
        lines.append(f"gross loss:      {gross_loss:.2f}")
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        lines.append(f"profit factor:   {pf:.2f}")
        lines.append(f"avg trade:       {df['pnl'].mean():.4f}")
        lines.append(f"best / worst:    {df['pnl'].max():.2f} / {df['pnl'].min():.2f}")
        lines.append("exit reasons:")
        for reason, n in df["reason"].value_counts().items():
            lines.append(f"  {reason:<12} {n:>6}   pnl {df[df['reason'] == reason]['pnl'].sum():.2f}")
        lines.append("grid usage:")
        for n, cnt in df["positions"].value_counts().sort_index().items():
            lines.append(f"  {n} position(s) {cnt:>6}")
    return "\n".join(lines), df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--balance", type=float, default=200.0)
    ap.add_argument("--lot", type=float, default=BASE_LOT)
    ap.add_argument("--point", type=float, default=0.01, help="broker point size (gold 2-digit = 0.01)")
    ap.add_argument("--contract-size", type=float, default=100.0, help="ounces per 1.00 lot")
    ap.add_argument("--spread-points", type=float, default=20.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--trades-out", default="trades.csv")
    args = ap.parse_args()

    bars = load_bars(args.csv)
    trades, equity, max_dd, _ = run(
        bars, args.balance, args.point, args.contract_size,
        args.spread_points, args.commission_per_lot, args.lot,
    )
    text, df = report(trades, args.balance, equity, max_dd, bars)
    print(text)
    if not df.empty:
        df.to_csv(args.trades_out, index=False)
        print(f"\ntrade list -> {args.trades_out}")


if __name__ == "__main__":
    main()
