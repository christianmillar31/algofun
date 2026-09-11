"""algofun command line.

    algofun fetch --universe sp500+etfs --start 1990-01-01
    algofun import-csv all_stocks_5yr.csv
    algofun list-strategies
    algofun backtest --strategy momentum --params lookback=126,top_n=10 --plot
    algofun walkforward --strategy sma_crossover
    algofun rebalance --strategy momentum --broker paper            # dry run
    algofun rebalance --strategy momentum --broker alpaca --execute  # paper account
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .backtest import COST_PRESETS, BacktestConfig, run_backtest, walk_forward
from .data import BarStore, import_long_csv, resolve_universe, sector_map
from .risk import DrawdownControl, RiskLimits
from .strategies import STRATEGIES, get_strategy, parse_params

log = logging.getLogger("algofun")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cache", default="data/cache", help="bar cache directory")
    p.add_argument("--universe", default=None,
                   help="sp500 | etfs | megacaps | a+b | AAPL,MSFT | path/to/file (default: everything cached)")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--min-bars", type=int, default=0, help="drop tickers with fewer cached bars")
    p.add_argument("--pit", action="store_true",
                   help="point-in-time: only hold names that were in the S&P 500 on each date (universe defaults to sp500-pit)")


def _add_backtest_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    p.add_argument("--params", default=None, help="k=v,k=v strategy params")
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--costs", default="retail", choices=sorted(COST_PRESETS))
    p.add_argument("--rebalance", default=None, help="override strategy schedule: daily|weekly|monthly|N")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--max-weight", type=float, default=0.25)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--max-sector", type=float, default=0.30, help="cap on any one sector's weight (needs sector map)")
    p.add_argument("--no-fractional", action="store_true", help="whole shares only")
    _add_drawdown_args(p)


def _add_drawdown_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dd-halve", type=float, default=0.10, help="halve exposure at this drawdown from peak")
    p.add_argument("--dd-flat", type=float, default=0.20, help="go flat at this drawdown from peak")
    p.add_argument("--no-drawdown-control", action="store_true", help="disable the drawdown budget")


def _drawdown_control(args):
    return None if args.no_drawdown_control else DrawdownControl(halve_at=args.dd_halve, flat_at=args.dd_flat)


def _load_panel(args):
    store = BarStore(args.cache)
    universe = args.universe or ("sp500-pit" if args.pit else None)
    tickers = resolve_universe(universe, cache_dir=Path(args.cache) / "universe") if universe else None
    panel = store.load_panel(tickers, min_bars=args.min_bars, sectors=sector_map(Path(args.cache) / "universe"))
    if args.pit:
        from .data import load_membership
        panel = panel.with_membership(load_membership(Path(args.cache) / "universe"))
        share = panel.membership.mean(axis=1).mean()
        print(f"point-in-time membership attached: on average {share:.0%} of the {len(panel.tickers)} loaded names are eligible per day")
    if len(panel) == 0:
        sys.exit("no bars in cache; run `algofun fetch` or `algofun import-csv` first")
    return store, panel


def _config(args) -> BacktestConfig:
    rb = args.rebalance
    if rb is not None and rb.isdigit():
        rb = int(rb)
    return BacktestConfig(
        initial_cash=args.cash, costs=COST_PRESETS[args.costs](),
        limits=RiskLimits(max_weight=args.max_weight, max_gross=args.max_gross, max_sector_weight=args.max_sector),
        allow_fractional=not args.no_fractional, rebalance=rb, drawdown_control=_drawdown_control(args),
        benchmark=args.benchmark or None,
    )


# ---- commands --------------------------------------------------------------
def cmd_fetch(args) -> None:
    store = BarStore(args.cache)
    tickers = resolve_universe(args.universe, cache_dir=Path(args.cache) / "universe")
    print(f"fetching {len(tickers)} tickers from {args.start} via {args.source} -> {store.bars_dir}")
    counts = store.update(tickers, start=args.start, end=args.end, source=args.source, force=args.force)
    ok = {t: n for t, n in counts.items() if n > 0}
    missing = sorted(t for t, n in counts.items() if n == 0)
    print(f"cached {len(ok)} tickers, {sum(ok.values()):,} bars total")
    if missing:
        print(f"no data for {len(missing)}: {', '.join(missing[:20])}{'...' if len(missing) > 20 else ''}")


def cmd_import_csv(args) -> None:
    store = BarStore(args.cache)
    counts = import_long_csv(args.path, store, ticker_col=args.ticker_col, date_col=args.date_col)
    print(f"imported {len(counts)} tickers, {sum(counts.values()):,} bars into {store.bars_dir}")


def cmd_list_strategies(args) -> None:
    for name, cls in sorted(STRATEGIES.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:<16} {doc}")
        print(f"{'':<16} defaults: {cls.defaults}")
        if cls.param_grid:
            print(f"{'':<16} grid:     {cls.param_grid}")


def cmd_backtest(args) -> None:
    _, panel = _load_panel(args)
    strat = get_strategy(args.strategy, **parse_params(args.params))
    cfg = _config(args)
    print(f"panel: {len(panel.tickers)} tickers, {len(panel)} bars "
          f"[{panel.dates[0].date()} -> {panel.dates[-1].date()}]  costs={args.costs}")
    res = run_backtest(panel, strat, cfg, start=args.start, end=args.end)
    print(res.summary())
    if args.stress:
        import dataclasses
        rows = [("1x", res.metrics())]
        for k in (2.0, 3.0):
            r_k = run_backtest(panel, get_strategy(args.strategy, **parse_params(args.params)),
                               dataclasses.replace(cfg, costs=cfg.costs.scaled(k)), start=args.start, end=args.end)
            rows.append((f"{k:.0f}x", r_k.metrics()))
        print("\nCost stress (an edge that dies at 2x was never there):")
        print(f"  {'costs':<6}{'cagr':>9}{'sharpe':>9}{'max_dd':>9}{'turnover':>10}")
        for label, m in rows:
            print(f"  {label:<6}{m['cagr']*100:8.2f}%{m['sharpe']:9.3f}{m['max_drawdown']*100:8.2f}%{m['annual_turnover']:9.1f}x")
    if args.out:
        res.save(args.out)
        print(f"saved to {args.out}/")
    if args.plot:
        _plot(res, args.out)


def cmd_walkforward(args) -> None:
    _, panel = _load_panel(args)
    cls = STRATEGIES[args.strategy]
    fixed = parse_params(args.params)
    grid = cls.param_grid
    if args.grid:
        grid = {k: [type(cls.defaults[k])(x) if k in cls.defaults else x for x in v.split("|")]
                for k, v in (item.split("=") for item in args.grid.split(","))}
    cfg = _config(args)
    print(f"panel: {len(panel.tickers)} tickers, {len(panel)} bars; grid={grid}; "
          f"train={args.train_bars} test={args.test_bars} objective={args.objective}")
    wf = walk_forward(panel.slice(args.start, args.end), cls, grid, train_bars=args.train_bars,
                      test_bars=args.test_bars, objective=args.objective, config=cfg, fixed_params=fixed,
                      verbose=args.verbose)
    print(wf.summary())
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        wf.folds.to_csv(out / "folds.csv", index=False)
        wf.equity.to_csv(out / "equity.csv")
        print(f"saved to {out}/")


def _make_broker(args):
    from .broker import get_broker
    if args.broker == "paper":
        return get_broker("paper", cash=args.cash, state_file=args.state_file,
                          costs=COST_PRESETS[args.costs]())
    import os
    paper_env = os.environ.get("ALPACA_PAPER", "true").strip().lower() != "false"
    if not paper_env and not args.live:
        sys.exit("ALPACA_PAPER=false but --live not given. Refusing to touch a live account by accident.")
    if args.live and paper_env:
        sys.exit("--live given but ALPACA_PAPER is not 'false'. Set both to trade live.")
    return get_broker("alpaca", paper=not args.live)


def _guardrails(args):
    from .live import Guardrails
    if args.no_guards:
        log.warning("GUARDS DISABLED by --no-guards")
        return Guardrails.off()
    return Guardrails(max_daily_loss_pct=args.max_daily_loss_pct, max_orders=args.max_orders,
                      max_order_pct=args.max_order_pct, max_stale_sessions=args.max_stale_sessions,
                      reconcile=not args.no_reconcile)


def cmd_rebalance(args) -> None:
    from .live import (
        execute_plan,
        load_state,
        plan_rebalance,
        run_guards,
        save_state,
        snapshot_state,
    )
    store = BarStore(args.cache)
    tickers = resolve_universe(args.universe, cache_dir=Path(args.cache) / "universe") if args.universe \
        else store.tickers()
    if args.refresh:
        print(f"refreshing bars for {len(tickers)} tickers...")
        store.update(tickers, start=args.start or "2000-01-01")
    strat = get_strategy(args.strategy, **parse_params(args.params))
    broker = _make_broker(args)
    sectors = sector_map(Path(args.cache) / "universe")
    if args.broker == "paper":
        panel = store.load_panel(tickers, min_bars=args.min_bars)
        broker.set_prices(panel.close.iloc[-1])
    limits = RiskLimits(max_weight=args.max_weight, max_gross=args.max_gross, max_sector_weight=args.max_sector)

    # ---- live gate: no real money until the paper record clears the bar ----
    if args.live and not args.override_gate:
        from .live import gate_status
        gs = gate_status(args.log, args.guard_log)
        print(gs.describe())
        if not gs.open:
            print("\nREFUSED: paper gate is closed. Keep paper trading, or pass --override-gate deliberately.")
            sys.exit(3)

    # ---- drawdown budget from the equity record (local log + broker history) ----
    acct0 = broker.account()
    history = _equity_history(args.equity_log, broker, acct0.equity)
    ctrl = _drawdown_control(args)
    scale, dd = ctrl.scale_from_history(history, acct0.equity) if ctrl else (1.0, 0.0)
    plan = plan_rebalance(strat, store, broker, tickers, limits=limits, min_bars=args.min_bars,
                          force=args.force, sectors=sectors, exposure_scale=scale, drawdown=dd)
    print(f"broker={broker.name}  strategy={strat.describe()}")
    print(plan.describe())

    # ---- kill switch: nothing below runs unless every guard passes ----
    acct = broker.account()
    positions = broker.positions()
    last_state = load_state(args.ledger)
    report = run_guards(plan, acct, positions, _guardrails(args), last_state=last_state)
    print(report.describe())
    _append_jsonl(args.guard_log, {"ts": _now_iso(), "as_of": str(plan.as_of.date()), "broker": broker.name,
                                   "passed": report.passed, "failures": report.failures, "orders": len(plan.orders),
                                   "execute": bool(args.execute), "equity": acct.equity})
    if not report.passed:
        print("\nBLOCKED: no orders submitted. Review the failures above; use --no-guards only deliberately.")
        sys.exit(2)

    if plan.skipped or not plan.orders:
        save_state(args.ledger, snapshot_state(broker, plan))
        return
    if not args.execute:
        print("\ndry run: nothing submitted (add --execute to send orders)")
        save_state(args.ledger, snapshot_state(broker, plan))
        return
    if args.broker != "paper":
        try:
            if not broker.is_market_open():
                print("market is closed: DAY orders will queue for the next open (this matches the backtest)")
        except Exception as e:  # noqa: BLE001 - informational only
            print(f"could not read the market clock ({e}); submitting anyway")
    results = execute_plan(plan, broker, log_path=args.log, settle_seconds=args.settle)
    for r in results:
        print(f"  {r.order.side:<4} {r.order.ticker:<6} {r.order.quantity:>10.4f}  {r.status}"
              f"{'  @ ' + format(r.filled_price, '.2f') if r.filled_price else ''}  {r.message}")
    acct = broker.account()
    print(f"account: cash={acct.cash:,.2f} equity={acct.equity:,.2f}")
    save_state(args.ledger, snapshot_state(broker, plan, results))
    print(f"state snapshot written to {args.ledger}")


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path, row: dict) -> None:
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _equity_history(path, broker, current_equity: float) -> list[float]:
    """Local daily equity record (one row per run day) merged with the broker's own history."""
    import json

    import pandas as pd
    rows = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                rows[str(pd.Timestamp(r["as_of"]).date())] = float(r["equity"])
            except (ValueError, KeyError):
                continue
    hist = broker.equity_history()
    if hist is not None:
        for d, v in hist.items():
            rows.setdefault(str(pd.Timestamp(d).date()), float(v))
    today = _now_iso()[:10]
    if today not in rows:
        _append_jsonl(path, {"as_of": today, "equity": current_equity, "ts": _now_iso()})
    rows[today] = current_equity
    return [rows[k] for k in sorted(rows)]


def cmd_gate(args) -> None:
    from .live import gate_status
    gs = gate_status(args.log, args.guard_log, min_fills=args.min_fills, min_sessions=args.min_sessions)
    print(gs.describe())
    sys.exit(0 if gs.open else 1)


def cmd_shortfall(args) -> None:
    from .live import shortfall_report
    rep = shortfall_report(args.log)
    if rep["n"] == 0:
        print("no fills with both a modelled and a realised price yet")
        return
    print(f"implementation shortfall over {rep['n']} fills:")
    print(f"  mean   {rep['mean_bps']:8.1f} bps   median {rep['median_bps']:8.1f} bps   total cost {rep['total_cost']:,.2f}")
    print("  worst five:")
    for r in rep["worst"]:
        print(f"    {r['as_of']} {r['side']:<4} {r['ticker']:<6} modelled {r['modelled']:9.2f} filled {r['filled']:9.2f}  {r['slip_bps']:7.1f} bps")


def cmd_flatten(args) -> None:
    """Emergency exit: cancel open orders and close every position."""
    broker = _make_broker(args)
    if args.broker == "paper":
        store = BarStore(args.cache)
        held = list(broker.positions())
        if held and all(store.has(t) for t in held):
            broker.set_prices(store.load_panel(held).close.iloc[-1])
    pos = broker.positions()
    acct = broker.account()
    print(f"broker={broker.name}  equity={acct.equity:,.2f}  positions={len(pos)}")
    for t, p in sorted(pos.items()):
        print(f"  {t:<6} qty={p.quantity:>10.4f} mv={p.market_value:>10.2f}")
    if not pos:
        print("nothing to flatten")
        return
    if not (args.execute and args.yes):
        print("\ndry run: pass BOTH --execute and --yes to close every position at market")
        return
    results = broker.flatten()
    for r in results:
        print(f"  {r.order.side:<4} {r.order.ticker:<6} {r.order.quantity:>10.4f}  {r.status}  {r.message}")
    acct = broker.account()
    print(f"account after: cash={acct.cash:,.2f} equity={acct.equity:,.2f}")


def cmd_status(args) -> None:
    broker = _make_broker(args)
    if args.broker == "paper":
        store = BarStore(args.cache)
        held = list(broker.positions())
        if held and all(store.has(t) for t in held):
            broker.set_prices(store.load_panel(held).close.iloc[-1])
    acct = broker.account()
    print(f"broker={broker.name}  cash={acct.cash:,.2f}  equity={acct.equity:,.2f}  "
          f"buying_power={acct.buying_power:,.2f}")
    pos = broker.positions()
    if not pos:
        print("no positions")
        return
    for t, p in sorted(pos.items()):
        print(f"  {t:<6} qty={p.quantity:>10.4f} avg={p.avg_price:>9.2f} mv={p.market_value:>10.2f}")


def _plot(res, out_dir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install 'algofun[plot]'")
        return
    from .backtest import drawdown_series
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    a1.plot(res.equity.index, res.equity, label=res.strategy)
    if res.benchmark is not None:
        a1.plot(res.benchmark.index, res.benchmark, label=res.benchmark.name, alpha=0.7)
    a1.set_ylabel("equity")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.fill_between(res.equity.index, drawdown_series(res.equity) * 100, 0, alpha=0.4)
    a2.set_ylabel("drawdown %")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(out_dir or "runs") / f"{res.strategy}_equity.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"plot saved to {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="algofun", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download bars into the cache")
    f.add_argument("--universe", default="sp500+etfs")
    f.add_argument("--start", default="1990-01-01")
    f.add_argument("--end", default=None)
    f.add_argument("--source", default="auto", choices=["auto", "yfinance", "stooq", "alpaca"])
    f.add_argument("--cache", default="data/cache")
    f.add_argument("--force", action="store_true", help="re-download full history")
    f.set_defaults(func=cmd_fetch)

    i = sub.add_parser("import-csv", help="bulk import a long-format CSV (date,open,high,low,close,volume,Name)")
    i.add_argument("path")
    i.add_argument("--ticker-col", default="Name")
    i.add_argument("--date-col", default="date")
    i.add_argument("--cache", default="data/cache")
    i.set_defaults(func=cmd_import_csv)

    ls = sub.add_parser("list-strategies", help="show strategies, defaults and grids")
    ls.set_defaults(func=cmd_list_strategies)

    b = sub.add_parser("backtest", help="run one strategy over the cached panel")
    _add_data_args(b)
    _add_backtest_args(b)
    b.add_argument("--out", default=None, help="directory to save equity/trades/metrics")
    b.add_argument("--plot", action="store_true")
    b.add_argument("--stress", action="store_true", help="also run at 2x and 3x the assumed costs")
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="rolling train/test parameter selection")
    _add_data_args(w)
    _add_backtest_args(w)
    w.add_argument("--grid", default=None, help="override grid: fast=20|50,slow=100|200")
    w.add_argument("--train-bars", type=int, default=504)
    w.add_argument("--test-bars", type=int, default=126)
    w.add_argument("--objective", default="sharpe")
    w.add_argument("--out", default=None)
    w.set_defaults(func=cmd_walkforward)

    for name, fn, help_ in (("rebalance", cmd_rebalance, "compute (and optionally send) today's orders"),
                            ("status", cmd_status, "show broker account and positions"),
                            ("flatten", cmd_flatten, "EMERGENCY: cancel open orders and close every position")):
        r = sub.add_parser(name, help=help_)
        r.add_argument("--broker", default="paper", choices=["paper", "alpaca"])
        r.add_argument("--cash", type=float, default=1_000.0, help="paper broker starting cash")
        r.add_argument("--state-file", default="paper_state.json", help="paper broker persistence")
        r.add_argument("--costs", default="retail", choices=sorted(COST_PRESETS))
        r.add_argument("--live", action="store_true", help="allow a LIVE Alpaca account (needs ALPACA_PAPER=false)")
        if name == "status":
            r.add_argument("--cache", default="data/cache")
        elif name == "flatten":
            r.add_argument("--cache", default="data/cache")
            r.add_argument("--execute", action="store_true", help="actually close positions")
            r.add_argument("--yes", action="store_true", help="second confirmation; required with --execute")
        else:
            _add_data_args(r)
            r.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
            r.add_argument("--params", default=None)
            r.add_argument("--max-weight", type=float, default=0.25)
            r.add_argument("--max-gross", type=float, default=1.0)
            r.add_argument("--max-sector", type=float, default=0.30, help="cap on any one sector's weight")
            r.add_argument("--refresh", action="store_true", help="fetch latest bars before deciding")
            r.add_argument("--execute", action="store_true", help="actually submit orders")
            r.add_argument("--force", action="store_true",
                           help="rebalance even if today is not a scheduled rebalance day")
            r.add_argument("--log", default="runs/rebalance_log.jsonl")
            g = r.add_argument_group("guards (kill switch)")
            g.add_argument("--ledger", default="runs/state.json",
                           help="position snapshot used for reconciliation between runs")
            g.add_argument("--max-daily-loss-pct", type=float, default=0.03,
                           help="block if equity is down more than this vs the prior close")
            g.add_argument("--max-orders", type=int, default=60, help="block if the plan has more orders than this")
            g.add_argument("--max-order-pct", type=float, default=0.30,
                           help="block if any single order exceeds this fraction of equity")
            g.add_argument("--max-stale-sessions", type=int, default=1,
                           help="block if the latest bar is older than this many trading days")
            g.add_argument("--no-reconcile", action="store_true", help="skip the position reconciliation check")
            g.add_argument("--no-guards", action="store_true", help="disable every guard (do not do this casually)")
            g.add_argument("--guard-log", default="runs/guard_log.jsonl")
            g.add_argument("--equity-log", default="runs/equity_log.jsonl")
            g.add_argument("--settle", type=float, default=0.0,
                           help="seconds to wait after submitting before refreshing fills for the log (Alpaca: 20)")
            g.add_argument("--override-gate", action="store_true",
                           help="allow --live even though the paper gate is closed")
            _add_drawdown_args(r)
        r.set_defaults(func=fn)

    gt = sub.add_parser("gate", help="is the paper record good enough to go live?")
    gt.add_argument("--log", default="runs/rebalance_log.jsonl")
    gt.add_argument("--guard-log", default="runs/guard_log.jsonl")
    gt.add_argument("--min-fills", type=int, default=100)
    gt.add_argument("--min-sessions", type=int, default=20)
    gt.set_defaults(func=cmd_gate)

    sf = sub.add_parser("shortfall", help="realised fills vs the prices the plan was sized at")
    sf.add_argument("--log", default="runs/rebalance_log.jsonl")
    sf.set_defaults(func=cmd_shortfall)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
