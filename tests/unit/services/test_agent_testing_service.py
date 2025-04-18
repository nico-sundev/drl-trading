from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai_trading.services.agent_testing_service import AgentTestingService


@pytest.fixture
def mock_env() -> MagicMock:
    env = MagicMock()
    env.reset.return_value = np.array([0, 0, 0, 0, 0])  # Mock observation
    env.step.return_value = (
        np.array([0, 0, 0, 0, 0]),  # observation
        0.0,  # reward
        False,  # done
        {},  # info
    )
    return env


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.predict = MagicMock(return_value=[0])
    return agent


@pytest.fixture
def mock_stock_data() -> Dict[str, List[float]]:
    return {"AAPL": [], "GOOGL": []}


def test_test_agent(
    mock_env: MagicMock, mock_agent: MagicMock, mock_stock_data: Dict[str, List[float]]
) -> None:
    # Arrange
    service = AgentTestingService()
    n_tests = 10

    # Act
    metrics = service.test_agent(
        mock_env, mock_agent, mock_stock_data, n_tests=n_tests, visualize=False
    )

    # Assert
    assert "steps" in metrics, "Metrics should include steps."
    assert "balances" in metrics, "Metrics should include balances."
    assert "net_worths" in metrics, "Metrics should include net worths."
    assert "shares_held" in metrics, "Metrics should include shares held."
    assert len(metrics["steps"]) == n_tests, "Number of steps should match n_tests."


@pytest.mark.skip(reason="Temporarily disabling this test")
def test_visualize_multiple_portfolio_net_worth() -> None:
    # Arrange
    service = AgentTestingService()
    steps = list(range(10))
    net_worths_list = [[100 + i for i in steps], [200 + i for i in steps]]
    labels = ["Agent1", "Agent2"]

    # Act & Assert
    # This test ensures no exceptions are raised during visualization
    service.visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels)
