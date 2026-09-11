"""algofun: an honest algorithmic trading lab.

Pipeline: data -> strategy -> backtest -> risk -> execution.
Everything is built so that the same Strategy object drives both the
backtester and the live/paper rebalancer.
"""

__version__ = "0.1.0"
