import importlib
import inspect
import pkgutil
from typing import Dict, Type

from ai_trading.agents.abstract_base_agent import AbstractBaseAgent
from ai_trading.agents.agent_policy import AgentPolicy
from ai_trading.agents.ensemble_agent import EnsembleAgent


class AgentRegistry:
    _instance = None

    def __new__(cls, *args, **kwargs) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self, package_name: str = "ai_trading.agents") -> None:
        if not hasattr(self, "agent_class_map"):
            self.package_name = package_name
            self.agent_class_map = self._discover_agents()

    def _discover_agents(self) -> Dict[str, Type[AbstractBaseAgent]]:
        agent_map: Dict[str, Type[AbstractBaseAgent]] = {}
        agent_package = importlib.import_module(self.package_name)

        for _, module_name, is_pkg in pkgutil.iter_modules(agent_package.__path__):
            if is_pkg or module_name in [
                "abstract_base_agent",
                "agent_policy",
                "agent_collection",
                "agent_registry",
                "agent_factory",
            ]:
                continue

            full_module_name = f"{self.package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Include all agent classes that inherit from AbstractBaseAgent but are not AgentPolicy itself
                if issubclass(obj, AbstractBaseAgent) and obj not in [
                    AbstractBaseAgent,
                    AgentPolicy,
                ]:
                    # Use lowercase name without "Agent" suffix as the key
                    agent_name = name.replace("Agent", "").lower()
                    agent_map[agent_name] = obj

        # Ensure EnsembleAgent is referenced by explicit name "ensemble" in the map
        if "ensemble" not in agent_map:
            agent_map["ensemble"] = EnsembleAgent

        return agent_map
