from .base import Account, Broker, Order, OrderResult, Position
from .paper import PaperBroker


def get_broker(name: str, **kwargs) -> Broker:
    name = name.lower()
    if name == "paper":
        return PaperBroker(**kwargs)
    if name == "alpaca":
        from .alpaca import AlpacaBroker
        return AlpacaBroker(**kwargs)
    raise ValueError(f"unknown broker {name!r}; choose paper or alpaca")


__all__ = ["Account", "Broker", "Order", "OrderResult", "PaperBroker", "Position", "get_broker"]
