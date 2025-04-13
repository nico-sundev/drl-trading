from ai_trading.agents.agent_registry import AgentRegistry
from ai_trading.agents.agent_collection import PPOAgent, A2CAgent, DDPGAgent, SACAgent, TD3Agent, EnsembleAgent
from typing import Dict, List
from ai_trading.trading_env import StockTradingEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from stable_baselines3.common.vec_env import DummyVecEnv


def create_env_and_train_agents(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    total_timesteps: int,
    threshold: float,
    agent_config: List[str],
):
    """
    Create environments and train agents dynamically based on the configuration.
    """
    # Create environments for training and validation
    train_env = DummyVecEnv([lambda: StockTradingEnv(train_data)])
    val_env = DummyVecEnv([lambda: StockTradingEnv(val_data)])
    
    agent_registry = AgentRegistry()

    agents = {}
    for agent_name in agent_config:
        if agent_name in agent_registry.agent_class_map:
            agent_class = agent_registry.agent_class_map[agent_name]
            agents[agent_name] = agent_class(train_env, total_timesteps, threshold)
            agents[agent_name].validate(val_env)

    # Create the ensemble agent if specified
    if "Ensemble" in agent_config:
        ensemble_agents = [agent for name, agent in agents.items() if name != "Ensemble"]
        agents["Ensemble"] = EnsembleAgent(ensemble_agents, threshold)
        agents["Ensemble"].validate(val_env)

    return train_env, val_env, agents


# -----------------------------------------------------------------------------


# Function to visualize portfolio changes
def visualize_portfolio(
    steps,
    balances,
    net_worths,
    shares_held,
    tickers,
    show_balance=True,
    show_net_worth=True,
    show_shares_held=True,
):

    fig, axs = plt.subplots(3, figsize=(12, 18))

    # Plot the balance
    if show_balance:
        axs[0].plot(steps, balances, label="Balance")
        axs[0].set_title("Balance Over Time")
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Balance")
        axs[0].legend()

    # Plot the net worth
    if show_net_worth:
        axs[1].plot(steps, net_worths, label="Net Worth", color="orange")
        axs[1].set_title("Net Worth Over Time")
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Net Worth")
        axs[1].legend()

    # Plot the shares held
    if show_shares_held:
        for ticker in tickers:
            axs[2].plot(steps, shares_held[ticker], label=f"Shares Held: {ticker}")
        axs[2].set_title("Shares Held Over Time")
        axs[2].set_xlabel("Steps")
        axs[2].set_ylabel("Shares Held")
        axs[2].legend()

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------


# function to visualize the portfolio net worth
def visualize_portfolio_net_worth(steps, net_worths):

    plt.figure(figsize=(12, 6))
    plt.plot(steps, net_worths, label="Net Worth", color="orange")
    plt.title("Net Worth Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------


# function to visualize the multiple portfolio net worths ( same chart )
def visualize_multiple_portfolio_net_worth(steps, net_worths_list, labels):

    plt.figure(figsize=(12, 6))
    for i, net_worths in enumerate(net_worths_list):
        plt.plot(steps, net_worths, label=labels[i])
    plt.title("Net Worth Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.show()


# -----------------------------------------------------------------------------


def test_agent(env, agent, stock_data, n_tests=1000, visualize=False):
    """Test a single agent and track performance metrics, with an option to visualize the results"""

    # Initialize metrics tracking
    metrics = {
        "steps": [],
        "balances": [],
        "net_worths": [],
        "shares_held": {ticker: [] for ticker in stock_data.keys()},
    }

    # Reset the environment before starting the tests
    obs = env.reset()

    for i in range(n_tests):

        metrics["steps"].append(i)

        action = agent.predict(obs)

        obs, rewards, dones, infos = env.step(action)

        if visualize:
            env.render()

        # Track metrics
        metrics["balances"].append(env.get_attr("balance")[0])
        metrics["net_worths"].append(env.get_attr("net_worth")[0])
        env_shares_held = env.get_attr("shares_held")[0]

        # Update shares held for each ticker
        for ticker in stock_data.keys():
            if ticker in env_shares_held:
                metrics["shares_held"][ticker].append(env_shares_held[ticker])
            else:
                metrics["shares_held"][ticker].append(
                    0
                )  # Append 0 if ticker is not found

        if dones:
            obs = env.reset()

    return metrics


# -----------------------------------------------------------------------------


def test_and_visualize_agents(env, agents, data, n_tests=1000):

    metrics = {}
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name}...")
        metrics[agent_name] = test_agent(
            env, agent, data, n_tests=n_tests, visualize=True
        )

    # Extract net worths for visualization
    net_worths = [metrics[agent_name]["net_worths"] for agent_name in agents.keys()]
    steps = next(iter(metrics.values()))[
        "steps"
    ]  # Assuming all agents have the same step count for simplicity

    # Visualize the performance metrics of multiple agents
    visualize_multiple_portfolio_net_worth(steps, net_worths, list(agents.keys()))


# -----------------------------------------------------------------------------


def compare_and_plot_agents(agents_metrics, labels, risk_free_rate=0.0):

    # Function to compare returns, standard deviation, and sharpe ratio of agents
    def compare_agents(agents_metrics, labels):
        returns = []
        stds = []
        sharpe_ratios = []

        for metrics in agents_metrics:

            net_worths = metrics["net_worths"]

            # Calculate daily returns
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

    # Compare agents
    df = compare_agents(agents_metrics, labels)

    # Sort the dataframe by sharpe ratio
    df_sorted = df.sort_values(by="Sharpe Ratio", ascending=False)

    # Display the dataframe
    display(df_sorted)

    # Plot bar chart for sharpe ratio
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted["Agent"], df_sorted["Sharpe Ratio"])
    plt.title("Sharpe Ratio Comparison")
    plt.xlabel("Agent")
    plt.ylabel("Sharpe Ratio")
    plt.show()
