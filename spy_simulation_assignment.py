"""
SPY Investment Simulation — course assignment (Parts 1–5, no bonus).

Downloads SPY adjusted closes, simulates $100/month DCA with strategies A/B/C,
reports metrics and plots, then analyzes missing the best SPY return days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# =============================================================================
# Part 1: Daily Returns
# =============================================================================
# Download historical daily adjusted close prices for SPY (e.g. Yahoo Finance)
# and compute daily simple returns: Return_t = (Price_t / Price_{t-1}) - 1.


def load_spy_adj_close(start: str, end: str) -> pd.Series:
    """Download SPY adjusted close as a timezone-naive daily Series."""
    raw = yf.download(
        "SPY",
        start=start,
        end=pd.to_datetime(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    col = raw["Adj Close"]
    if isinstance(col, pd.DataFrame):
        col = col.squeeze(axis=1)
    s = col.dropna().sort_index()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s


def daily_returns_from_prices(prices: pd.Series) -> pd.Series:
    """Part 1: daily simple returns from adjusted close."""
    return prices.pct_change().dropna()


# =============================================================================
# Part 2: Investment Simulation (shared helpers)
# =============================================================================
# Assumptions: $100 per month, fractional shares, Jan 1993 – Mar 30, 2026,
# invest at adjusted close on the chosen day; shares += 100 / price.
#
# Timing convention: on an investment day, the existing portfolio grows by the
# day's return first, then $100 is added. The new cash therefore earns 0%
# on its first day, which is consistent with buying at the day's close.


def _trading_days_in_month(
    prices: pd.Series, month_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DatetimeIndex:
    mask = (
        (prices.index >= month_start)
        & (prices.index.year == month_start.year)
        & (prices.index.month == month_start.month)
        & (prices.index <= end_date)
    )
    return prices.index[mask]


def _first_trading_day_of_month(
    prices: pd.Series, month_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Timestamp | None:
    days = _trading_days_in_month(prices, month_start, end_date)
    if len(days) == 0:
        return None
    return days[0]


def _first_weekday_trading_day_in_month(
    prices: pd.Series,
    month_start: pd.Timestamp,
    end_date: pd.Timestamp,
    weekday: int,
) -> pd.Timestamp | None:
    """First trading day in the calendar month that falls on `weekday` (Mon=0)."""
    for d in _trading_days_in_month(prices, month_start, end_date):
        if d.weekday() == weekday:
            return d
    return None


def invest_date_strategy_a(
    prices: pd.Series, month_start: pd.Timestamp, end_date: pd.Timestamp
) -> pd.Timestamp | None:
    """Part 3 Strategy A: first trading day of the month."""
    return _first_trading_day_of_month(prices, month_start, end_date)


def invest_date_strategy_b_or_c(
    prices: pd.Series,
    month_start: pd.Timestamp,
    end_date: pd.Timestamp,
    target_weekday: int,
) -> pd.Timestamp | None:
    """
    Part 3 Strategy B/C: first trading Mon/Wed of the month only if within
    5 calendar days after the month's first trading day; else first trading day.

    delta_days == 0 is intentionally included: if the first trading day itself
    is the target weekday, we invest on that day (satisfies "within 5 days").
    """
    first_trading = _first_trading_day_of_month(prices, month_start, end_date)
    if first_trading is None:
        return None

    target_day = _first_weekday_trading_day_in_month(
        prices, month_start, end_date, target_weekday
    )
    if target_day is None:
        return first_trading

    delta_days = (target_day.normalize() - first_trading.normalize()).days
    if 0 <= delta_days <= 5:
        return target_day
    return first_trading


def monthly_investment_dates(
    prices: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategy: str,
) -> list[pd.Timestamp]:
    """All scheduled investment dates for one strategy (chronological)."""
    strategy = strategy.strip().upper()

    if strategy == "A":
        def picker(ms):
            return invest_date_strategy_a(prices, ms, end_date)
    elif strategy == "B":
        def picker(ms):
            return invest_date_strategy_b_or_c(prices, ms, end_date, 0)
    elif strategy == "C":
        def picker(ms):
            return invest_date_strategy_b_or_c(prices, ms, end_date, 2)
    else:
        raise ValueError('strategy must be "A", "B", or "C"')

    dates: list[pd.Timestamp] = []
    for nominal in pd.date_range(start=start_date, end=end_date, freq="MS"):
        d = picker(nominal)
        if d is not None and d <= end_date:
            dates.append(pd.Timestamp(d))
    return sorted(dates)


def simulate_daily_portfolio_value(
    prices: pd.Series,
    invest_dates: set[pd.Timestamp],
    monthly_amount: float,
    miss_return_dates: set[pd.Timestamp] | None = None,
) -> pd.Series:
    """
    Part 2 (day-by-day): NAV with optional Part 5 rule — on dates in
    miss_return_dates, apply 0% daily return instead of SPY's realized return.

    Timing: existing portfolio grows by the day's return, then the $100 flow
    is added. New cash earns 0% on its first day (equivalent to buying at close).
    """
    miss_return_dates = miss_return_dates or set()
    idx = prices.index.sort_values()
    nav = pd.Series(index=idx, dtype=float)
    prev_nav = 0.0

    for i, day in enumerate(idx):
        p = float(prices.loc[day])
        if i == 0:
            growth = 1.0
        elif day in miss_return_dates:
            growth = 1.0
        else:
            p_prev = float(prices.loc[idx[i - 1]])
            growth = p / p_prev if p_prev else 1.0

        flow = monthly_amount if day in invest_dates else 0.0
        cur = prev_nav * growth + flow
        nav.loc[day] = cur
        prev_nav = cur

    return nav


# =============================================================================
# Part 3: Investment Strategies (schedules only; simulation above)
# =============================================================================
# Strategy A: first trading day each month.
# Strategy B: first trading Monday within 5 calendar days after first trading
#             day; else first trading day.
# Strategy C: same as B for Wednesday.


# =============================================================================
# Part 4: Output — totals, profit, IRR, plots
# =============================================================================


def xirr(
    amounts: list[float],
    dates: list[datetime],
    guess: float = 0.1,
) -> float | None:
    """
    Annualized IRR from signed cash flows at irregular dates (Newton + bisection).
    amounts: negative for outflows, positive for inflows.
    """
    if len(amounts) < 2 or any(math.isnan(x) for x in amounts):
        return None

    t0 = dates[0]
    years = [(d - t0).days / 365.25 for d in dates]

    def npv(rate: float) -> float:
        return sum(a / (1.0 + rate) ** y for a, y in zip(amounts, years))

    r = guess
    for _ in range(50):
        f = npv(r)
        if abs(f) < 1e-8:
            return r
        eps = 1e-6
        df = (npv(r + eps) - f) / eps
        if df == 0:
            break
        r_new = r - f / df
        if r_new <= -0.9999 or r_new > 1e6 or math.isnan(r_new):
            break
        r = r_new

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None

    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = npv(mid)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


@dataclass
class StrategySummary:
    strategy: str
    total_invested: float
    total_shares: float
    final_value: float
    profit: float
    annualized_irr: float | None


def summarize_strategy(
    strategy: str,
    invest_dates: list[pd.Timestamp],
    final_nav: float,
    monthly_amount: float,
    valuation_date: pd.Timestamp,
    prices: pd.Series,
) -> StrategySummary:
    """Part 4: totals, final value, profit, IRR from dated cash flows."""
    total_invested = monthly_amount * len(invest_dates)
    total_shares = sum(
        monthly_amount / float(prices.loc[d]) for d in invest_dates
    )
    profit = final_nav - total_invested

    amounts: list[float] = []
    flow_dates: list[datetime] = []
    for d in invest_dates:
        amounts.append(-monthly_amount)
        flow_dates.append(pd.Timestamp(d).to_pydatetime())
    amounts.append(final_nav)
    flow_dates.append(pd.Timestamp(valuation_date).to_pydatetime())

    irr_val = xirr(amounts, flow_dates)
    return StrategySummary(
        strategy=strategy,
        total_invested=total_invested,
        total_shares=total_shares,
        final_value=final_nav,
        profit=profit,
        annualized_irr=irr_val,
    )


def plot_portfolio_comparison(
    nav_by_strategy: dict[str, pd.Series], outfile: str
) -> None:
    """Part 4: portfolio value over time and visual comparison of strategies."""
    plt.figure(figsize=(10, 5))
    for label, series in sorted(nav_by_strategy.items()):
        plt.plot(series.index, series.values, label=f"Strategy {label}")
    plt.title("SPY monthly DCA — portfolio value over time (adjusted close)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


# =============================================================================
# Part 5: Extension — missing the best SPY market days
# =============================================================================
# For each strategy, if the investor "misses" the top K daily return days
# (SPY), assume 0% portfolio return on those calendar dates; compare final
# balance, profit, and annualized IRR to baseline.


def best_days_dates(daily_ret: pd.Series, k: int) -> set[pd.Timestamp]:
    """Dates with the K largest one-day SPY simple returns."""
    if k <= 0:
        return set()
    top = daily_ret.nlargest(k)
    return set(pd.Timestamp(t) for t in top.index)


# =============================================================================
# Print summary helpers
# =============================================================================


def print_part1_daily_returns_summary(prices: pd.Series, daily_ret: pd.Series) -> None:
    """Console output for Part 1: data source, range, and return sample/stats."""
    print("=" * 72)
    print("Part 1: Daily Returns")
    print("=" * 72)
    print("  Source: Yahoo Finance via yfinance (ticker SPY).")
    print("  Price field: Adjusted Close.")
    print(
        "  Formula: Return_t = (Price_t / Price_{t-1}) - 1  (simple daily return)"
    )
    first_p, last_p = prices.index.min(), prices.index.max()
    print(f"  Adjusted close sample: {first_p.date()}  to  {last_p.date()}")
    print(f"  Trading days (prices): {len(prices):,}")
    print(f"  Trading days (returns): {len(daily_ret):,}")
    print("  Daily return summary (full sample):")
    print(f"    mean: {daily_ret.mean():.6f}    std: {daily_ret.std():.6f}")
    print(f"    min:  {daily_ret.min():.6f}    max: {daily_ret.max():.6f}")
    print("  First 3 daily returns:")
    for idx, r in daily_ret.head(3).items():
        print(f"    {pd.Timestamp(idx).date()}  {r:+.6f}")
    print("  Last 3 daily returns:")
    for idx, r in daily_ret.tail(3).items():
        print(f"    {pd.Timestamp(idx).date()}  {r:+.6f}")
    print()


def print_part2_simulation_assumptions(
    monthly_amount: float, start: str, end: str
) -> None:
    """Console output for Part 2: DCA assumptions."""
    print("=" * 72)
    print("Part 2: Investment Simulation")
    print("=" * 72)
    print(f"  Fixed contribution each month: ${monthly_amount:,.2f}")
    print("  Fractional shares: allowed.")
    print(f"  Investment window: {start}  through  {end} (inclusive).")
    print("  Execution price: adjusted close on the scheduled investment day.")
    print("  Shares purchased each time: monthly_amount / price_on_that_day.")
    print("  Tracked over time: total shares accumulated and portfolio value (NAV).")
    print()


def print_part3_strategy_definitions() -> None:
    """Console output for Part 3: strategies A, B, and C."""
    print("=" * 72)
    print("Part 3: Investment Strategies")
    print("=" * 72)
    print("  Strategy A: Invest on the first trading day of each month.")
    print(
        "  Strategy B: Invest on the first trading Monday of the month if it falls "
        "within 5 calendar days after that month's first trading day; "
        "otherwise invest on the first trading day."
    )
    print(
        "  Strategy C: Same as B, but using the first trading Wednesday "
        "instead of Monday."
    )
    print()


def print_part5_table(rows: list[dict]) -> None:
    """Print comparison table for Part 5 (baseline vs miss top days)."""
    headers = ["Case", "Final Balance", "Total Investment", "Profit", "Annualized Return"]
    col_w = [28, 16, 18, 14, 20]

    def fmt_money(x: float) -> str:
        return f"${x:,.2f}"

    def fmt_pct(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{x * 100:.2f}%"

    line = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(line)
    print("-" * len(line))
    for r in rows:
        vals = [
            r["case"],
            fmt_money(r["final_balance"]),
            fmt_money(r["total_investment"]),
            fmt_money(r["profit"]),
            fmt_pct(r["annualized_return"]),
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(vals, col_w)))


def main() -> None:
    start = "1993-01-01"
    end = "2026-03-30"
    monthly_amount = 100.0
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # -------------------------------------------------------------------------
    # Part 1: Daily Returns
    # -------------------------------------------------------------------------
    prices = load_spy_adj_close(start, end)
    prices = prices[prices.index <= end_ts]
    daily_ret = daily_returns_from_prices(prices)
    print_part1_daily_returns_summary(prices, daily_ret)

    # -------------------------------------------------------------------------
    # Part 2: Simulation assumptions (execution in loop below)
    # -------------------------------------------------------------------------
    print_part2_simulation_assumptions(monthly_amount, start, end)

    # -------------------------------------------------------------------------
    # Part 3: Strategy definitions (schedules applied in loop below)
    # -------------------------------------------------------------------------
    print_part3_strategy_definitions()

    # -------------------------------------------------------------------------
    # Parts 2–4 continued: simulate A/B/C, metrics, plots
    # -------------------------------------------------------------------------
    # Compute last trading day once — shared by all strategies and Part 5.
    last_day = prices.index[prices.index <= end_ts].max()

    # Cache invest lists/sets and NAVs per strategy so Part 5 can reuse them
    # without re-running date selection or re-simulating the baseline.
    strategy_invest_lists: dict[str, list[pd.Timestamp]] = {}
    strategy_invest_sets: dict[str, set[pd.Timestamp]] = {}
    nav_by_strategy: dict[str, pd.Series] = {}
    summaries: list[StrategySummary] = []

    for strat in ("A", "B", "C"):
        invest_list = monthly_investment_dates(prices, start_ts, end_ts, strat)
        invest_set = set(invest_list)
        strategy_invest_lists[strat] = invest_list
        strategy_invest_sets[strat] = invest_set

        nav = simulate_daily_portfolio_value(
            prices, invest_set, monthly_amount, miss_return_dates=None
        )
        nav_by_strategy[strat] = nav

        final_nav = float(nav.loc[last_day])
        summaries.append(
            summarize_strategy(
                strat, invest_list, final_nav, monthly_amount, last_day, prices
            )
        )

    print("=" * 72)
    print("Part 4: Output")
    print("=" * 72)
    print("  Metrics and IRR as of the last trading day in the sample.")
    print("  Portfolio value over time: see comparison plot file below.")
    print()
    for s in summaries:
        irr_s = "n/a" if s.annualized_irr is None else f"{s.annualized_irr * 100:.2f}%"
        print(f"Strategy {s.strategy}")
        print(f"  Total amount invested:    ${s.total_invested:,.2f}")
        print(f"  Total shares accumulated:   {s.total_shares:.4f}")
        print(f"  Final portfolio value:    ${s.final_value:,.2f}")
        print(f"  Total profit:             ${s.profit:,.2f}")
        print(f"  Annualized return (IRR):  {irr_s}")
        print()

    plot_path = "spy_simulation_portfolio_comparison.png"
    plot_portfolio_comparison(nav_by_strategy, plot_path)
    print(f"  Saved comparison chart (strategies A/B/C): {plot_path!r}")
    print()

    # -------------------------------------------------------------------------
    # Part 5: Miss best 5 / 10 / 15 / 20 SPY return days
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("Part 5: Extension - Missing the best SPY market days")
    print("=" * 72)
    print(
        "  On the top-K SPY daily return days (sample-wide), portfolio daily "
        "return is set to 0% instead of the market return; compare to baseline."
    )
    print()

    miss_ks = (5, 10, 15, 20)

    for strat in ("A", "B", "C"):
        print("-" * 72)
        print(f"Part 5 - Strategy {strat}: baseline vs. missing best return days")
        print("-" * 72)

        # Reuse cached dates and baseline NAV from Part 2–4 (no re-computation).
        invest_list = strategy_invest_lists[strat]
        invest_set = strategy_invest_sets[strat]
        base_final = float(nav_by_strategy[strat].loc[last_day])
        base_summary = summarize_strategy(
            strat, invest_list, base_final, monthly_amount, last_day, prices
        )

        rows: list[dict] = [
            {
                "case": "Baseline",
                "final_balance": base_final,
                "total_investment": base_summary.total_invested,
                "profit": base_summary.profit,
                "annualized_return": base_summary.annualized_irr,
            }
        ]

        for k in miss_ks:
            miss_dates = best_days_dates(daily_ret, k)
            nav_miss = simulate_daily_portfolio_value(
                prices, invest_set, monthly_amount, miss_return_dates=miss_dates
            )
            final_miss = float(nav_miss.loc[last_day])
            summ = summarize_strategy(
                strat, invest_list, final_miss, monthly_amount, last_day, prices
            )
            rows.append(
                {
                    "case": f"Miss Best {k} Days",
                    "final_balance": final_miss,
                    "total_investment": summ.total_invested,
                    "profit": summ.profit,
                    "annualized_return": summ.annualized_irr,
                }
            )

        print_part5_table(rows)
        print()

    print("=" * 72)
    print("Artifacts written")
    print("=" * 72)
    print(f"  Part 4 portfolio comparison plot: {plot_path!r}")
    print()


if __name__ == "__main__":
    main()