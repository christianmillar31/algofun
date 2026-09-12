"""The paper-trading gate: count trades and clean sessions, not calendar days.

Live mode refuses to run until the paper record clears the gate, unless the
operator overrides it explicitly. Thresholds follow the practitioner rules of
thumb gathered in the research: at least 100 filled orders and 20
consecutive runs in which every guard passed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FILLED_STATUSES = {"filled", "partially_filled"}


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@dataclass
class GateStatus:
    fills: int
    fill_sessions: int
    first_fill: str | None
    last_fill: str | None
    clean_sessions: int          # consecutive guard-passing runs at the tail of the guard log
    total_runs: int
    min_fills: int
    min_sessions: int
    notes: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.fills >= self.min_fills and self.clean_sessions >= self.min_sessions

    def describe(self) -> str:
        lines = [f"paper gate: {'OPEN' if self.open else 'CLOSED'}",
                 f"  filled paper orders      {self.fills:6d}  (need {self.min_fills})",
                 f"  sessions with fills      {self.fill_sessions:6d}  ({self.first_fill or '-'} to {self.last_fill or '-'})",
                 f"  consecutive clean runs   {self.clean_sessions:6d}  (need {self.min_sessions}; {self.total_runs} runs logged)"]
        lines += [f"  note  {n}" for n in self.notes]
        return "\n".join(lines)


def gate_status(fill_log: str | Path, guard_log: str | Path, min_fills: int = 100, min_sessions: int = 20,
                paper_only: bool = True) -> GateStatus:
    fills = [r for r in _read_jsonl(fill_log)
             if (not paper_only or "paper" in str(r.get("broker", "")))
             and (r.get("status") in FILLED_STATUSES or float(r.get("filled_quantity") or 0) > 0)]
    sessions = sorted({str(r.get("as_of")) for r in fills})
    guard_rows = [r for r in _read_jsonl(guard_log) if not paper_only or "paper" in str(r.get("broker", ""))]
    clean = 0
    for r in reversed(guard_rows):
        if r.get("passed"):
            clean += 1
        else:
            break
    notes = []
    if not fills:
        notes.append("no paper fills recorded yet")
    if not guard_rows:
        notes.append("no guard results recorded yet (runs before the guard log existed do not count)")
    return GateStatus(fills=len(fills), fill_sessions=len(sessions), first_fill=sessions[0] if sessions else None,
                      last_fill=sessions[-1] if sessions else None, clean_sessions=clean, total_runs=len(guard_rows),
                      min_fills=min_fills, min_sessions=min_sessions, notes=notes)


def shortfall_report(fill_log: str | Path) -> dict:
    """Implementation shortfall from the fill log: realised fill vs the price the
    plan was sized at. Positive bps = paid more than modelled."""
    rows = []
    for r in _read_jsonl(fill_log):
        fp, mp, q = r.get("filled_price"), r.get("modelled_price"), float(r.get("filled_quantity") or 0)
        if not fp or not mp or q <= 0:
            continue
        sign = 1.0 if r.get("side") == "buy" else -1.0
        bps = sign * (float(fp) / float(mp) - 1.0) * 1e4
        rows.append({"as_of": r.get("as_of"), "broker": r.get("broker"), "ticker": r.get("ticker"), "side": r.get("side"),
                     "qty": q, "modelled": float(mp), "filled": float(fp), "slip_bps": bps,
                     "cost": sign * (float(fp) - float(mp)) * q})
    if not rows:
        return {"n": 0, "rows": []}
    bps = sorted(x["slip_bps"] for x in rows)
    mid = len(bps) // 2
    median = bps[mid] if len(bps) % 2 else (bps[mid - 1] + bps[mid]) / 2
    return {"n": len(rows), "mean_bps": sum(bps) / len(bps), "median_bps": median,
            "total_cost": sum(x["cost"] for x in rows), "worst": sorted(rows, key=lambda x: -x["slip_bps"])[:5],
            "rows": rows}
