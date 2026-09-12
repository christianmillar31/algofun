"""Kill switch and sanity checks that run before any order is sent.

The rules here are the retail-scale version of the pre-trade controls every
regulated automated-trading desk is required to have: a data-freshness gate,
a daily loss limit, order count and size caps, and a reconciliation of what
the broker says we hold against what we last recorded. Any failure blocks
the run and exits non-zero so the scheduler turns red and a human looks.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..broker.base import Account, Broker, OrderResult, Position
from .calendar import previous_trading_day
from .rebalance import RebalancePlan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Guardrails:
    max_daily_loss_pct: float = 0.03      # halt if equity is down more than this vs the prior close
    max_orders: int = 60                  # per run
    max_order_pct: float = 0.30           # any single order's notional as a fraction of equity
    max_gross_order_pct: float = 1.10     # sum of all order notionals as a fraction of equity
    max_stale_sessions: int = 1           # latest bar may be at most this many trading days old
    reconcile: bool = True                # compare broker positions with the last recorded state
    reconcile_tolerance: float = 0.01     # relative quantity drift allowed on untouched positions

    @classmethod
    def off(cls) -> Guardrails:
        return cls(max_daily_loss_pct=1.0, max_orders=10**9, max_order_pct=1e9, max_gross_order_pct=1e9,
                   max_stale_sessions=10**6, reconcile=False)


@dataclass
class GuardReport:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        lines = [f"guards: {'PASS' if self.passed else 'BLOCKED'}"]
        lines += [f"  FAIL  {f}" for f in self.failures]
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)


def _today(now: datetime | None) -> pd.Timestamp:
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        now = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001 - no tz database: fall back to UTC
        pass
    return pd.Timestamp(now.date())


def check_freshness(as_of: pd.Timestamp, max_stale_sessions: int = 1, now: datetime | None = None) -> str | None:
    """Return a failure message if the latest bar is too old (or from the future)."""
    today = _today(now)
    as_of = pd.Timestamp(as_of).normalize()
    if as_of > today:
        return f"latest bar {as_of.date()} is after today {today.date()}: clock or data problem"
    if max_stale_sessions > 260:   # more than a trading year: the check is effectively disabled
        return None
    allowed = today
    for _ in range(max_stale_sessions):
        allowed = previous_trading_day(allowed)
    if as_of < allowed:
        return (f"latest bar {as_of.date()} is stale: expected at least {allowed.date()} "
                f"(max {max_stale_sessions} session(s) old)")
    return None


def check_daily_loss(account: Account, max_daily_loss_pct: float) -> tuple[str | None, str | None]:
    """(failure, note). Uses the broker's prior-close equity when it provides one."""
    if account.last_equity is None or account.last_equity <= 0:
        return None, "no prior-close equity from broker; daily loss limit not checked"
    pnl = account.equity / account.last_equity - 1.0
    if pnl < -max_daily_loss_pct:
        return f"daily loss {pnl:.2%} exceeds limit {max_daily_loss_pct:.2%} (equity {account.equity:,.2f} vs {account.last_equity:,.2f})", None
    return None, f"day P&L {pnl:+.2%} within limit {max_daily_loss_pct:.0%}"


def check_order_limits(plan: RebalancePlan, equity: float, g: Guardrails) -> list[str]:
    fails = []
    if len(plan.orders) > g.max_orders:
        fails.append(f"{len(plan.orders)} orders exceeds max {g.max_orders}")
    if equity <= 0:
        if plan.orders:
            fails.append("equity is zero or negative; refusing to trade")
        return fails
    gross = 0.0
    for o in plan.orders:
        px = float(plan.prices.get(o.ticker, float("nan")))
        if pd.isna(px) or px <= 0:
            fails.append(f"no reference price for {o.ticker}")
            continue
        notional = o.quantity * px
        gross += notional
        if notional > g.max_order_pct * equity:
            fails.append(f"{o.side} {o.ticker} notional {notional:,.0f} is {notional / equity:.0%} of equity "
                         f"(max {g.max_order_pct:.0%})")
    if gross > g.max_gross_order_pct * equity:
        fails.append(f"gross order notional {gross:,.0f} is {gross / equity:.0%} of equity (max {g.max_gross_order_pct:.0%})")
    return fails


def check_reconciliation(positions: dict[str, Position], last_state: dict | None,
                         tolerance: float = 0.01) -> tuple[list[str], str | None]:
    """Compare broker positions with the last recorded snapshot.

    Names that were part of the previous run's orders are expected to differ
    (fills complete after the snapshot) and are exempt.
    """
    if not last_state:
        return [], "no reconciliation baseline yet (first run); snapshot will be written after this run"
    recorded: dict[str, float] = {k: float(v) for k, v in last_state.get("positions", {}).items()}
    pending = set(last_state.get("pending", []))
    fails = []
    for t, p in positions.items():
        if t in pending:
            continue
        if t not in recorded:
            fails.append(f"unexpected position {t} qty {p.quantity:g} not in last recorded state")
            continue
        base = max(abs(recorded[t]), 1e-9)
        if abs(p.quantity - recorded[t]) / base > tolerance:
            fails.append(f"{t} qty {p.quantity:g} differs from recorded {recorded[t]:g}")
    for t, q in recorded.items():
        if t in pending or t in positions:
            continue
        if abs(q) > 1e-9:
            fails.append(f"recorded position {t} qty {q:g} is no longer held")
    return fails, f"reconciled {len(positions)} positions against snapshot from {last_state.get('as_of', '?')}"


def run_guards(plan: RebalancePlan, account: Account, positions: dict[str, Position], g: Guardrails,
               last_state: dict | None = None, now: datetime | None = None) -> GuardReport:
    r = GuardReport()
    f = check_freshness(plan.as_of, g.max_stale_sessions, now)
    if f:
        r.failures.append(f)
    else:
        r.notes.append(f"latest bar {pd.Timestamp(plan.as_of).date()} is fresh")
    f, n = check_daily_loss(account, g.max_daily_loss_pct)
    if f:
        r.failures.append(f)
    if n:
        r.notes.append(n)
    r.failures += check_order_limits(plan, account.equity, g)
    if g.reconcile:
        fails, note = check_reconciliation(positions, last_state, g.reconcile_tolerance)
        r.failures += fails
        if note:
            r.notes.append(note)
    return r


# ---- state ledger ----------------------------------------------------------
def snapshot_state(broker: Broker, plan: RebalancePlan, results: list[OrderResult] | None = None) -> dict:
    positions = broker.positions()
    acct = broker.account()
    pending = sorted({r.order.ticker for r in (results or []) if r.status not in ("rejected",)})
    return {
        "ts": datetime.now(timezone.utc).isoformat(), "broker": broker.name, "as_of": str(pd.Timestamp(plan.as_of).date()),
        "equity": acct.equity, "cash": acct.cash,
        "positions": {t: p.quantity for t, p in positions.items()},
        "pending": pending,
    }


def load_state(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        log.warning("state file %s is unreadable; treating as no baseline", p)
        return None


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))
