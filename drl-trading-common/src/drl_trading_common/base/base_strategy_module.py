from injector import Module
from typing import Protocol

class BaseStrategyModule(Protocol):
    def as_injector_module(self) -> Module:
        ...
