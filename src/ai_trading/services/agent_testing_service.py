import matplotlib.pyplot as plt
import numpy as np

class AgentTestingService:
    """
    Service to handle testing and visualization of agents.
    """

    def test_agent(self, env, agent, stock_data, n_tests=1000, visualize=False):
        """
        Test a single agent and track performance metrics, with an option to visualize the results.

        Args:
            env: The environment to test the agent in.
            agent: The agent to be tested.
            stock_data: The stock data used for testing.
            n_tests (int): Number of test iterations.
            visualize (bool): Whether to visualize the environment during testing.

        Returns:
            dict: Metrics collected during testing.
        """
        metrics = {
            "steps": [],
            "balances": [],
            "net_worths": [],
            "shares_held": {ticker: [] for ticker in stock_data.keys()},
        }

        obs = env.reset()

        for i in range(n_tests):
            metrics["steps"].append(i)

            action = agent.predict(obs)
            obs, rewards, dones, infos = env.step(action)

            if visualize:
                env.render()

            metrics["balances"].append(env.get_attr("balance")[0])
            metrics["net_worths"].append(env.get_attr("net_worth")[0])
            env_shares_held = env.get_attr("shares_held")[0]

            for ticker in stock_data.keys():
                metrics["shares_held"][ticker].append(env_shares_held.get(ticker, 0))

            if dones:
                obs = env.reset()

        return metrics

    def visualize_multiple_portfolio_net_worth(self, steps, net_worths_list, labels):
        """
        Visualize the net worths of multiple agents on the same chart.

        Args:
            steps: The steps during testing.
            net_worths_list: List of net worths for each agent.
            labels: Labels for each agent.
        """
        plt.figure(figsize=(12, 6))
        for i, net_worths in enumerate(net_worths_list):
            plt.plot(steps, net_worths, label=labels[i])
        plt.title("Net Worth Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.show()

    def test_and_visualize_agents(self, env, agents, data, n_tests=1000):
        """
        Test and visualize the performance of multiple agents.

        Args:
            env: The environment to test the agents in.
            agents: A dictionary of agents to be tested.
            data: The stock data used for testing.
            n_tests (int): Number of test iterations.
        """
        metrics = {}
        for agent_name, agent in agents.items():
            print(f"Testing {agent_name}...")
            metrics[agent_name] = self.test_agent(env, agent, data, n_tests=n_tests, visualize=True)

        net_worths = [metrics[agent_name]["net_worths"] for agent_name in agents.keys()]
        steps = next(iter(metrics.values()))["steps"]

        self.visualize_multiple_portfolio_net_worth(steps, net_worths, list(agents.keys()))