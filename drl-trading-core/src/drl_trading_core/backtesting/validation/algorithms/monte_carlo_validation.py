"""
Monte Carlo Validation for trading strategies.

This module provides Monte Carlo simulation to assess the statistical robustness
of trading strategy performance, helping distinguish skill from luck.
"""

import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.monte_carlo_config import (
    MonteCarloConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class MonteCarloValidation(BaseValidationAlgorithm[MonteCarloConfig]):
    """
    Validates a trading strategy using Monte Carlo simulations.

    This validator runs thousands of simulations with randomized performance paths
    to assess whether the strategy's performance is likely due to skill rather than luck.
    It provides statistical confidence intervals and probability of various outcomes.

    Different simulation methods are supported:
    - Returns bootstrap: Randomly resample returns (assumes independence)
    - Block bootstrap: Resample blocks of returns (preserves some autocorrelation)
    - Parametric: Generate synthetic returns from a fitted distribution
    """

    def __init__(self, config: MonteCarloConfig):
        super().__init__(config)
        self.name = "MonteCarloValidation"

        # Validate configuration
        if self.config.num_simulations < 100:
            raise ValueError(
                f"num_simulations should be at least 100, got {self.config.num_simulations}"
            )

        if not 0 < self.config.confidence_level < 100:
            raise ValueError(
                f"confidence_level must be between 0 and 100, got {self.config.confidence_level}"
            )

        if (
            self.config.simulation_method == "block_bootstrap"
            and self.config.block_length < 1
        ):
            raise ValueError(
                f"block_length must be positive, got {self.config.block_length}"
            )

        if not 0 <= self.config.min_acceptable_win_rate <= 100:
            raise ValueError(
                f"min_acceptable_win_rate must be between 0 and 100, got {self.config.min_acceptable_win_rate}"
            )

        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Run Monte Carlo simulations to validate strategy robustness.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome and Monte Carlo analysis.
        """
        # Get the equity curve and calculate returns
        equity = strategy.get_equity_curve()

        if equity.empty:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.config.min_acceptable_win_rate,
                explanation="No equity data available for Monte Carlo simulation.",
            )

        # Calculate daily returns
        returns = equity.pct_change().dropna()

        if len(returns) < 30:  # Need a minimum sample size for meaningful simulations
            return ValidationResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.config.min_acceptable_win_rate,
                explanation=f"Insufficient data for Monte Carlo simulation. Got {len(returns)} data points, need at least 30.",
            )

        # Run Monte Carlo simulations based on the configured method
        sim_paths, sim_stats = self._run_simulations(returns, equity.iloc[0])

        # Calculate confidence intervals
        lower_bound, median, upper_bound = self._calculate_confidence_intervals(
            sim_paths
        )

        # Calculate win rate (percentage of simulations that end with positive returns)
        win_rate = self._calculate_win_rate(sim_paths)

        # Calculate additional metrics
        metrics = self._calculate_additional_metrics(returns, sim_paths, sim_stats)

        # Generate visualization if requested
        visualization_base64 = self._generate_visualization(
            equity, sim_paths, lower_bound, median, upper_bound
        )

        # Determine if validation passed based on win rate
        passed = win_rate >= self.config.min_acceptable_win_rate

        # Create detailed explanation
        explanation = self._create_explanation(
            passed, win_rate, sim_stats, metrics, visualization_base64
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=win_rate,
            threshold=self.config.min_acceptable_win_rate,
            explanation=explanation,
        )

    def _run_simulations(
        self, returns: pd.Series, initial_equity: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run Monte Carlo simulations using the configured method.

        Args:
            returns: Historical returns series
            initial_equity: Initial equity value

        Returns:
            Tuple containing:
            - Array of simulated equity paths
            - Dictionary of simulation statistics
        """
        num_days = len(returns)
        sim_returns = np.zeros((self.config.num_simulations, num_days))

        if self.config.simulation_method == "returns_bootstrap":
            # Simple bootstrap by randomly sampling from historical returns
            for i in range(self.config.num_simulations):
                sim_returns[i] = np.random.choice(
                    returns.values, size=num_days, replace=True
                )

        elif self.config.simulation_method == "block_bootstrap":
            # Block bootstrap to preserve some of the autocorrelation
            block_length = self.config.block_length

            for i in range(self.config.num_simulations):
                # Generate blocks until we have enough days
                sim_data: List[float] = []
                while len(sim_data) < num_days:
                    # Randomly select a starting point for the block
                    start_idx = np.random.randint(0, len(returns) - block_length + 1)
                    # Extract the block
                    block = returns.values[start_idx : start_idx + block_length]
                    sim_data.extend(block)

                # Trim to exact length
                sim_returns[i] = np.array(sim_data[:num_days])

        elif self.config.simulation_method == "parametric":
            # Parametric approach - fit a distribution to returns and sample from it
            mean = returns.mean()
            std = returns.std()

            # Generate random returns from a normal distribution with same mean and std
            for i in range(self.config.num_simulations):
                sim_returns[i] = np.random.normal(mean, std, size=num_days)

        # Convert returns to equity paths
        sim_paths = np.cumprod(1 + sim_returns, axis=1) * initial_equity

        # Calculate basic simulation statistics
        final_values = sim_paths[:, -1]
        sim_stats = {
            "mean": np.mean(final_values),
            "median": np.median(final_values),
            "min": np.min(final_values),
            "max": np.max(final_values),
            "std": np.std(final_values),
        }

        return sim_paths, sim_stats

    def _calculate_confidence_intervals(
        self, sim_paths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for each time point across all simulations.

        Args:
            sim_paths: Array of simulated equity paths

        Returns:
            Tuple of lower bound, median, and upper bound arrays
        """
        alpha = (100 - self.config.confidence_level) / 100
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(sim_paths, lower_percentile, axis=0)
        median = np.percentile(sim_paths, 50, axis=0)
        upper_bound = np.percentile(sim_paths, upper_percentile, axis=0)

        return lower_bound, median, upper_bound

    def _calculate_win_rate(self, sim_paths: np.ndarray) -> float:
        """
        Calculate the win rate as the percentage of simulations that end positive.

        Args:
            sim_paths: Array of simulated equity paths

        Returns:
            Win rate as a percentage
        """
        # Get the final value of each simulation
        final_values = sim_paths[:, -1]

        # Compare with the initial values
        initial_values = sim_paths[:, 0]

        # Calculate percentage of simulations that ended with a profit
        profitable_sims = np.sum(final_values > initial_values)
        win_rate = (profitable_sims / self.config.num_simulations) * 100

        return float(win_rate)

    def _calculate_additional_metrics(
        self, original_returns: pd.Series, sim_paths: np.ndarray, sim_stats: Dict
    ) -> Dict:
        """
        Calculate additional performance metrics across simulations.

        Args:
            original_returns: Original strategy returns
            sim_paths: Array of simulated equity paths
            sim_stats: Dictionary of basic simulation statistics

        Returns:
            Dictionary of additional metrics
        """
        metrics = {}

        # Calculate maximum drawdown distribution
        max_drawdowns = []
        for i in range(sim_paths.shape[0]):
            path = sim_paths[i, :]
            peak = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak * 100
            max_drawdowns.append(np.min(drawdown))

        metrics["max_drawdown_mean"] = np.mean(max_drawdowns)
        metrics["max_drawdown_median"] = np.median(max_drawdowns)
        metrics["max_drawdown_worst"] = np.min(max_drawdowns)

        # Calculate original strategy metrics for comparison
        original_cumulative_return = (1 + original_returns).cumprod().iloc[-1] - 1
        metrics["original_return"] = original_cumulative_return * 100  # as percentage

        # Calculate the percentile rank of the original strategy in the simulations
        final_returns = (sim_paths[:, -1] / sim_paths[:, 0]) - 1
        metrics["strategy_percentile"] = np.percentile(
            np.sort(final_returns) * 100,  # as percentages
            np.searchsorted(np.sort(final_returns), original_cumulative_return)
            / len(final_returns)
            * 100,
        )

        # Add probability of loss
        metrics["probability_of_loss"] = 100 - self._calculate_win_rate(sim_paths)

        return metrics

    def _generate_visualization(
        self,
        original_equity: pd.Series,
        sim_paths: np.ndarray,
        lower_bound: np.ndarray,
        median: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Optional[str]:
        """
        Generate a visualization of Monte Carlo simulations.

        Args:
            original_equity: Original strategy equity curve
            sim_paths: Array of simulated equity paths
            lower_bound: Lower confidence bound
            median: Median path
            upper_bound: Upper confidence bound

        Returns:
            Base64 encoded string of the plot image or None if visualization fails
        """
        try:
            # Create the figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot a subset of simulated paths (plotting all would be too cluttered)
            num_to_plot = min(100, self.config.num_simulations)
            subset_indices = np.random.choice(
                sim_paths.shape[0], num_to_plot, replace=False
            )

            for i in subset_indices:
                ax.plot(sim_paths[i, :], "lightgray", alpha=0.3)

            # Plot the original equity curve
            ax.plot(
                original_equity.values, "blue", linewidth=2, label="Original Strategy"
            )

            # Plot confidence intervals
            x = range(len(lower_bound))
            ax.plot(x, median, "r--", linewidth=2, label="Median")
            ax.plot(
                x,
                lower_bound,
                "g-",
                linewidth=1.5,
                label=f"{self.config.confidence_level}% Confidence Interval",
            )
            ax.plot(x, upper_bound, "g-", linewidth=1.5)
            ax.fill_between(x, lower_bound, upper_bound, color="g", alpha=0.1)

            # Set labels and title
            ax.set_title("Monte Carlo Simulation of Equity Curve")
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Equity")
            ax.legend()

            # Convert plot to base64 string
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            return img_str

        except Exception as _e:
            # If visualization fails, return None (visualization is optional)
            return None

    def _create_explanation(
        self,
        passed: bool,
        win_rate: float,
        sim_stats: Dict,
        metrics: Dict,
        visualization_base64: Optional[str],
    ) -> str:
        """
        Create a detailed explanation based on Monte Carlo analysis.

        Args:
            passed: Whether the validation passed
            win_rate: Percentage of simulations with positive returns
            sim_stats: Dictionary of simulation statistics
            metrics: Dictionary of additional metrics
            visualization_base64: Base64 encoded visualization or None

        Returns:
            Detailed explanation string with optional embedded visualization
        """
        explanation = f"Monte Carlo simulation with {self.config.num_simulations} paths using '{self.config.simulation_method}' method.\n\n"

        if passed:
            explanation += f"Validation PASSED - Win rate of {win_rate:.2f}% exceeds minimum threshold of {self.config.min_acceptable_win_rate:.2f}%.\n"
        else:
            explanation += f"Validation FAILED - Win rate of {win_rate:.2f}% is below minimum threshold of {self.config.min_acceptable_win_rate:.2f}%.\n"

        # Add simulation statistics
        explanation += "\nSimulation Results:\n"
        explanation += f"- Win Rate (% of profitable simulations): {win_rate:.2f}%\n"
        explanation += f"- Mean Final Equity: {sim_stats['mean']:.2f}\n"
        explanation += f"- Median Final Equity: {sim_stats['median']:.2f}\n"
        explanation += f"- Range: {sim_stats['min']:.2f} to {sim_stats['max']:.2f}\n"

        # Add additional metrics
        explanation += "\nRisk Metrics:\n"
        explanation += f"- Probability of Loss: {metrics['probability_of_loss']:.2f}%\n"
        explanation += (
            f"- Average Maximum Drawdown: {metrics['max_drawdown_mean']:.2f}%\n"
        )
        explanation += f"- Worst Case Drawdown: {metrics['max_drawdown_worst']:.2f}%\n"

        # Add strategy comparison
        explanation += "\nStrategy Performance Evaluation:\n"
        explanation += (
            f"- Original Strategy Return: {metrics['original_return']:.2f}%\n"
        )
        explanation += f"- Strategy Percentile Rank: {metrics['strategy_percentile']:.2f}th percentile of simulations\n"

        # Add confidence intervals
        explanation += f"\nAt {self.config.confidence_level:.1f}% confidence level, we can expect the strategy performance to fall within the visualized range.\n"

        # Add statistical interpretation
        if win_rate > 95:
            explanation += "\nStatistical Interpretation: The strategy shows VERY STRONG evidence of positive expected returns with high statistical confidence.\n"
        elif win_rate > 80:
            explanation += "\nStatistical Interpretation: The strategy shows STRONG evidence of positive expected returns.\n"
        elif win_rate > 65:
            explanation += "\nStatistical Interpretation: The strategy shows MODERATE evidence of positive expected returns.\n"
        elif win_rate > 50:
            explanation += "\nStatistical Interpretation: The strategy shows WEAK evidence of positive expected returns.\n"
        else:
            explanation += "\nStatistical Interpretation: The strategy does NOT show evidence of positive expected returns. The results are more consistent with randomness or negative expectancy.\n"

        # Add visualization if available
        if visualization_base64:
            explanation += (
                "\n\n![Monte Carlo Simulation](data:image/png;base64,"
                + visualization_base64
                + ")"
            )

        return explanation


# Register the validation algorithm
register_validation("MonteCarloValidation", MonteCarloValidation, MonteCarloConfig)
