import importlib
import inspect
import pkgutil
from typing import Dict, Type

from ai_trading.agents.agent_collection import PPOAgent


class AgentRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, package_name="ai_trading.agents"):
        if not hasattr(self, "agent_class_map"):
            self.package_name = package_name
            self.agent_class_map = self._discover_agents()

    def _discover_agents(self) -> Dict[str, Type[PPOAgent]]:
        agent_map = {}
        agent_package = importlib.import_module(self.package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(agent_package.__path__):
            if is_pkg:
                continue

            full_module_name = f"{self.package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, PPOAgent) and obj is not PPOAgent:
                    agent_name = name.replace("Agent", "").lower()
                    agent_map[agent_name] = obj

        # Ensure EnsembleAgent is always included
        from ai_trading.agents.agent_collection import EnsembleAgent

        agent_map["EnsembleAgent"] = EnsembleAgent

        return agent_map
