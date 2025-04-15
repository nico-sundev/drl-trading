import pytest
from unittest.mock import MagicMock
from ai_trading.services.agent_testing_service import AgentTestingService

@pytest.fixture
def mock_env():
    return MagicMock()

@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.predict = MagicMock(return_value=[0])
    return agent

@pytest.fixture
def mock_stock_data():
    return {"AAPL": [], "GOOGL": []}

def test_test_agent(mock_env, mock_agent, mock_stock_data):
    # Arrange
    service = AgentTestingService()
    n_tests = 10

    # Act
    metrics = service.test_agent(mock_env, mock_agent, mock_stock_data, n_tests=n_tests, visualize=False)

    # Assert
    assert "steps" in metrics, "Metrics should include steps."
    assert "balances" in metrics, "Metrics should include balances."
    assert "net_worths" in metrics, "Metrics should include net worths."
    assert "shares_held" in metrics, "Metrics should include shares held."
    assert len(metrics["steps"]) == n_tests, "Number of steps should match n_tests."

def test_visualize_multiple_portfolio_net_worth():
    # Arrange
    service = AgentTestingService()
    steps = list(range(10))
    net_worths_list = [[100 + i for i in steps], [200 + i for i in steps]]
    labels = ["Agent1", "Agent2"]

    # Act & Assert
    # This test ensures no exceptions are raised during visualization
    service.visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels)