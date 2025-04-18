import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from ai_trading.services.agent_testing_service import AgentTestingService


class TestAndVisualize:
    def __init__(self, agent_testing_service: AgentTestingService):
        self.agent_testing_service = agent_testing_service

    def test_and_visualize_agents(
        self, env, agents: dict, data: pd.DataFrame, n_tests: int
    ):
        self.agent_testing_service.test_and_visualize_agents(
            env, agents, data, n_tests=n_tests
        )

    def visualize_portfolio(
        self,
        steps: list,
        balances: list,
        net_worths: list,
        shares_held: dict,
        tickers: list,
        show_balance: bool = True,
        show_net_worth: bool = True,
        show_shares_held: bool = True,
    ):
        fig, axs = plt.subplots(3, figsize=(12, 18))

        if show_balance:
            axs[0].plot(steps, balances, label="Balance")
            axs[0].set_title("Balance Over Time")
            axs[0].set_xlabel("Steps")
            axs[0].set_ylabel("Balance")
            axs[0].legend()

        if show_net_worth:
            axs[1].plot(steps, net_worths, label="Net Worth", color="orange")
            axs[1].set_title("Net Worth Over Time")
            axs[1].set_xlabel("Steps")
            axs[1].set_ylabel("Net Worth")
            axs[1].legend()

        if show_shares_held:
            for ticker in tickers:
                axs[2].plot(steps, shares_held[ticker], label=f"Shares Held: {ticker}")
            axs[2].set_title("Shares Held Over Time")
            axs[2].set_xlabel("Steps")
            axs[2].set_ylabel("Shares Held")
            axs[2].legend()

        plt.tight_layout()
        plt.show()

    def visualize_portfolio_net_worth(self, steps: list, net_worths: list):
        plt.figure(figsize=(12, 6))
        plt.plot(steps, net_worths, label="Net Worth", color="orange")
        plt.title("Net Worth Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.show()

    def visualize_multiple_portfolio_net_worth(
        self, steps: list, net_worths_list: list, labels: list
    ):
        plt.figure(figsize=(12, 6))
        for i, net_worths in enumerate(net_worths_list):
            plt.plot(steps, net_worths, label=labels[i])
        plt.title("Net Worth Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.show()

    def compare_and_plot_agents(
        self, agents_metrics: list, labels: list, risk_free_rate: float = 0.0
    ):
        def compare_agents(agents_metrics, labels):
            returns = []
            stds = []
            sharpe_ratios = []

            for metrics in agents_metrics:
                net_worths = metrics["net_worths"]
                daily_returns = np.diff(net_worths) / net_worths[:-1]
                avg_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                sharpe_ratio = (
                    ((avg_return - risk_free_rate) / std_return)
                    if std_return != 0
                    else "Inf"
                )

                returns.append(avg_return)
                stds.append(std_return)
                sharpe_ratios.append(sharpe_ratio)

            df = pd.DataFrame(
                {
                    "Agent": labels,
                    "Return": returns,
                    "Standard Deviation": stds,
                    "Sharpe Ratio": sharpe_ratios,
                }
            )

            return df

        df = compare_agents(agents_metrics, labels)
        df_sorted = df.sort_values(by="Sharpe Ratio", ascending=False)
        display(df_sorted)

        plt.figure(figsize=(12, 6))
        plt.bar(df_sorted["Agent"], df_sorted["Sharpe Ratio"])
        plt.title("Sharpe Ratio Comparison")
        plt.xlabel("Agent")
        plt.ylabel("Sharpe Ratio")
        plt.show()
