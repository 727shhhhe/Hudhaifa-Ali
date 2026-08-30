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
    def __init__(self, direction: int, price: float, volume: float, bar: int = 0,
                 sl_distance: float = 0.0, tp_distance=None):
        self.direction = direction
        self.entries = [(price, volume)]
        self.peak_profit = 0.0
        self.open_bar = bar
        self.sl_distance = sl_distance  # price distance, for non-cash stop modes
        self.tp_distance = tp_distance  # price distance, None disables the TP

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


def atr_series(bars: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def run(bars: pd.DataFrame, balance: float, point: float, contract_size: float,
        spread_points: float, commission_per_lot: float, lot: float = BASE_LOT,
        sl_mode: str = "cash", sl_cash: float = HARD_STOP_CASH,
        sl_points: float = 22.0, atr_mult: float = 1.0, atr_period: int = 14,
        max_hold_bars: int = 0, stop_on_blowup: bool = False,
        tp_mode: str = "points", tp_points: float = TP_POINTS,
        tp_atr_mult: float = 1.0, tp_rr: float = 1.0,
        max_positions: int = MAX_POSITIONS,
        grid_step_points: float = GRID_STEP_POINTS,
        trail_activate: float = TRAIL_ACTIVATE,
        trail_retrace: float = TRAIL_RETRACE):
    """sl_mode: 'cash' (fixed cash cap, as specified), 'points' (price distance
    from the basket average), 'atr' (atr_mult x ATR at entry) or 'none'
    (TP/trail only). max_hold_bars > 0 adds a time stop.

    tp_mode: 'points' (fixed distance from basket average), 'atr'
    (tp_atr_mult x ATR at entry), 'rr' (tp_rr x the stop distance) or 'none'
    (trailing lock / stop only)."""
    spread = spread_points * point
    need_atr = sl_mode == "atr" or tp_mode == "atr"
    atr = atr_series(bars, atr_period).to_numpy() if need_atr else None
    equity = balance
    basket = None
    trades = []
    equity_curve = []
    peak_equity = balance
    max_dd = 0.0
    peak_float_equity = balance
    max_float_dd = 0.0
    min_float_equity = balance
    margin_call_at = None
    floating = 0.0

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

                if sl_mode == "cash":
                    distance = sl_cash / vol_value
                elif sl_mode == "none":
                    distance = None
                else:
                    distance = basket.sl_distance

                if distance is not None:
                    sl_price = avg - distance if long_side else avg + distance
                    if (mark <= sl_price) if long_side else (mark >= sl_price):
                        close_basket(fill(sl_price), t[i], "hard_stop")
                        continue

                if trail_activate > 0:
                    if floating >= trail_activate:
                        basket.peak_profit = max(basket.peak_profit, floating)
                    if (basket.peak_profit >= trail_activate
                            and basket.peak_profit - floating >= trail_retrace):
                        close_basket(mark, t[i], "trail_lock")
                        continue

                if basket.tp_distance is not None:
                    tp_price = avg + basket.tp_distance if long_side \
                        else avg - basket.tp_distance
                    if (mark >= tp_price) if long_side else (mark <= tp_price):
                        close_basket(fill(tp_price), t[i], "take_profit")
                        continue

                if basket.count < max_positions:
                    grid = grid_step_points * point
                    grid_price = basket.last_price - grid if long_side \
                        else basket.last_price + grid
                    if (mark <= grid_price) if long_side else (mark >= grid_price):
                        basket.add(fill(grid_price), lot)

            if (basket is not None and max_hold_bars > 0
                    and i - basket.open_bar >= max_hold_bars):
                exit_price = c[i] if basket.direction == BUY else c[i] + spread
                close_basket(exit_price, t[i], "time_stop")

        floating = 0.0
        if basket is not None:
            mark_close = c[i] if basket.direction == BUY else c[i] + spread
            floating = basket.profit(mark_close, contract_size)
        float_equity = equity + floating
        peak_float_equity = max(peak_float_equity, float_equity)
        max_float_dd = max(max_float_dd, peak_float_equity - float_equity)
        if float_equity < min_float_equity:
            min_float_equity = float_equity
        if margin_call_at is None and float_equity <= 0.0:
            margin_call_at = t[i]
        equity_curve.append((t[i], equity, float_equity))

        if stop_on_blowup and float_equity <= 0.0:
            break

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

        if sl_mode == "points":
            sl_distance = sl_points * point
        elif sl_mode == "atr":
            sl_distance = atr[i] * atr_mult
            if not (sl_distance > 0):
                continue
        elif sl_mode == "cash":
            # cash stop: distance depends on live basket volume, resolved per tick
            sl_distance = sl_cash / (lot * contract_size)
        else:
            sl_distance = 0.0

        if tp_mode == "points":
            tp_distance = tp_points * point
        elif tp_mode == "atr":
            tp_distance = atr[i] * tp_atr_mult
            if not (tp_distance > 0):
                continue
        elif tp_mode == "rr":
            if sl_distance <= 0:
                continue
            tp_distance = sl_distance * tp_rr
        else:
            tp_distance = None

        entry_bid = o[i + 1]
        if mom > 0 and mom_slope >= 0:
            basket = Basket(BUY, entry_bid + spread, lot, i + 1, sl_distance, tp_distance)
        elif mom < 0 and mom_slope <= 0:
            basket = Basket(SELL, entry_bid, lot, i + 1, sl_distance, tp_distance)

    stats = {
        "open_floating": floating if basket is not None else 0.0,
        "float_equity": equity + (floating if basket is not None else 0.0),
        "max_realized_dd": max_dd,
        "max_float_dd": max_float_dd,
        "min_float_equity": min_float_equity,
        "margin_call_at": margin_call_at,
    }
    return trades, equity, max_dd, equity_curve, stats


def report(trades, start_balance, equity, max_dd, bars, stats=None):
    df = pd.DataFrame(trades)
    lines = []
    lines.append(f"bars:            {len(bars)}  ({bars['datetime'].iloc[0]} -> {bars['datetime'].iloc[-1]})")
    lines.append(f"start balance:   {start_balance:.2f}")
    lines.append(f"end equity:      {equity:.2f}")
    lines.append(f"net profit:      {equity - start_balance:.2f}")
    lines.append(f"max drawdown:    {max_dd:.2f}   (realized)")
    if stats:
        lines.append(f"open floating:   {stats['open_floating']:.2f}")
        lines.append(f"equity + float:  {stats['float_equity']:.2f}")
        lines.append(f"max float DD:    {stats['max_float_dd']:.2f}")
        lines.append(f"min float equity:{stats['min_float_equity']:.2f}")
        lines.append(f"margin call:     {stats['margin_call_at']}")
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
    ap.add_argument("--sl-mode", choices=("cash", "points", "atr", "none"), default="cash")
    ap.add_argument("--sl-cash", type=float, default=HARD_STOP_CASH)
    ap.add_argument("--sl-points", type=float, default=22.0)
    ap.add_argument("--atr-mult", type=float, default=1.0)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--max-hold-bars", type=int, default=0)
    ap.add_argument("--stop-on-blowup", action="store_true")
    ap.add_argument("--tp-mode", choices=("points", "atr", "rr", "none"), default="points")
    ap.add_argument("--tp-points", type=float, default=TP_POINTS)
    ap.add_argument("--tp-atr-mult", type=float, default=1.0)
    ap.add_argument("--tp-rr", type=float, default=1.0)
    ap.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    ap.add_argument("--grid-step-points", type=float, default=GRID_STEP_POINTS)
    ap.add_argument("--trail-activate", type=float, default=TRAIL_ACTIVATE)
    ap.add_argument("--trail-retrace", type=float, default=TRAIL_RETRACE)
    ap.add_argument("--trades-out", default="trades.csv")
    args = ap.parse_args()

    bars = load_bars(args.csv)
    trades, equity, max_dd, _, stats = run(
        bars, args.balance, args.point, args.contract_size,
        args.spread_points, args.commission_per_lot, args.lot,
        args.sl_mode, args.sl_cash, args.sl_points, args.atr_mult,
        args.atr_period, args.max_hold_bars, args.stop_on_blowup,
        args.tp_mode, args.tp_points, args.tp_atr_mult, args.tp_rr,
        args.max_positions, args.grid_step_points,
        args.trail_activate, args.trail_retrace,
    )
    text, df = report(trades, args.balance, equity, max_dd, bars, stats)
    print(text)
    if not df.empty:
        df.to_csv(args.trades_out, index=False)
        print(f"\ntrade list -> {args.trades_out}")


if __name__ == "__main__":
    main()
