import types
from unittest import mock

import pytest

from ai_trading.agents.agent_registry import AgentRegistry
from ai_trading.agents.ppo_agent import PPOAgent


# Mocked Agent classes for testing
class MockPPOAgent(PPOAgent):
    pass


class MockA2CAgent(PPOAgent):
    pass


@pytest.fixture(autouse=True)
def registry():
    return AgentRegistry(package_name="tests.unit.agents")


# Test the discover_agents function
@pytest.mark.slow
def test_discover_agents(registry):
    with mock.patch("pkgutil.iter_modules") as mock_iter_modules, mock.patch(
        "importlib.import_module"
    ) as mock_import_module:

        # Use the package name of the current test file
        # registry = AgentRegistry(package_name="tests.unit.agents")

        # Mock the base package (with __path__) and an agent module
        mock_base_package = mock.Mock()
        mock_base_package.__path__ = ["path_to_agents"]

        # Create mock module dynamically with real class objects
        mock_agent_module = types.ModuleType("mock_agent_module")
        mock_agent_module.MockPPOAgent = MockPPOAgent
        mock_agent_module.MockA2CAgent = MockA2CAgent

        def import_module_side_effect(name):
            if name == "ai_trading.agents":
                return mock_base_package
            else:
                return mock_agent_module

        mock_import_module.side_effect = import_module_side_effect

        # Mock iter_modules to simulate one module in the test file's path
        mock_iter_modules.return_value = [
            ("path_to_agents", "agent_registry_test", False)
        ]

        # Now when registry accesses .agent_class_map, it will discover those agents
        agent_map = registry.agent_class_map

        assert "mockppo" in agent_map
        assert "mocka2c" in agent_map
        assert agent_map["mockppo"].__name__ == "MockPPOAgent"
        assert agent_map["mocka2c"].__name__ == "MockA2CAgent"
        assert len(agent_map) == 3  # Includes EnsembleAgent
        assert "EnsembleAgent" in agent_map
