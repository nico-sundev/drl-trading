"""
Statistical Significance Validation for trading strategies.

This module provides formal hypothesis testing to determine if a strategy's performance
is statistically significant or could be explained by chance alone.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from drl_trading_core.backtesting.registry import register_validation
from drl_trading_core.backtesting.strategy.strategy_interface import (
    StrategyInterface,
)
from drl_trading_core.backtesting.validation.algorithms.base_validation_algorithm import (
    BaseValidationAlgorithm,
)
from drl_trading_core.backtesting.validation.algorithms.config.statistical_significance_config import (
    StatisticalSignificanceConfig,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class StatisticalSignificanceValidation(
    BaseValidationAlgorithm[StatisticalSignificanceConfig]
):
    """
    Validates a trading strategy through statistical hypothesis testing.

    This validator applies formal statistical methods to determine if a strategy's
    performance can be distinguished from chance. It supports multiple testing
    methods and corrections for multiple comparisons.

    The validator can test various null hypotheses:
    - Zero mean (strategy returns are not significantly different from zero)
    - Random walk (strategy returns are not distinguishable from a random walk)
    - Benchmark comparison (strategy does not significantly outperform a benchmark)
    """

    def __init__(self, config: StatisticalSignificanceConfig):
        super().__init__(config)
        self.name = "StatisticalSignificanceValidation"

        # Validate configuration
        if not 0 < self.config.significance_level < 1:
            raise ValueError(
                f"significance_level must be between 0 and 1, got {self.config.significance_level}"
            )

        if self.config.min_sample_size < 10:
            raise ValueError(
                f"min_sample_size should be at least 10, got {self.config.min_sample_size}"
            )

        if (
            self.config.null_hypothesis == "benchmark"
            and self.config.benchmark_symbol is None
        ):
            raise ValueError(
                "benchmark_symbol must be specified when null_hypothesis is 'benchmark'"
            )

        # Supported statistical tests
        self.supported_tests = {
            "t_test",
            "bootstrap",
            "wilcoxon",
            "jarque_bera",
            "shapiro",
            "runs_test",
            "white_reality_check",
        }

        # Check that all requested tests are supported
        unsupported = set(self.config.tests) - self.supported_tests
        if unsupported:
            raise ValueError(
                f"Unsupported test(s): {unsupported}. Supported tests: {self.supported_tests}"
            )

    def run(self, strategy: StrategyInterface) -> ValidationResult:
        """
        Run statistical significance tests on the strategy's performance.

        Args:
            strategy: The strategy to validate.

        Returns:
            ValidationResult containing the validation outcome and statistical analysis.
        """
        # Get data based on return type configuration
        if self.config.return_type == "time":
            # Get equity curve and calculate returns
            equity = strategy.get_equity_curve()

            if equity.empty:
                return ValidationResult(
                    name=self.name,
                    passed=False,
                    score=1.0,  # p-value of 1 = no significance
                    threshold=self.config.significance_level,
                    explanation="No equity data available for statistical testing.",
                )

            returns = equity.pct_change().dropna()

        else:  # trade returns
            trades = strategy.get_trades(include_open=False)

            if not trades:
                return ValidationResult(
                    name=self.name,
                    passed=False,
                    score=1.0,  # p-value of 1 = no significance
                    threshold=self.config.significance_level,
                    explanation="No completed trades available for statistical testing.",
                )

            # Extract returns from trade profits
            # We need to normalize by trade size/risk for comparability
            returns = pd.Series(
                [
                    (
                        trade.profit / (trade.size * trade.entry_price)
                        if trade.entry_price > 0 and trade.size > 0
                        else trade.profit
                    )
                    for trade in trades
                ]
            )

        # Check if we have enough data
        if len(returns) < self.config.min_sample_size:
            return ValidationResult(
                name=self.name,
                passed=False,
                score=1.0,
                threshold=self.config.significance_level,
                explanation=(
                    f"Insufficient data for statistical significance testing. "
                    f"Got {len(returns)} data points, need at least {self.config.min_sample_size}."
                ),
            )

        # Run the requested statistical tests
        test_results = self._run_statistical_tests(returns, strategy)

        # Apply multiple testing correction if more than one test
        if len(test_results) > 1 and self.config.correction_method != "none":
            self._apply_multiple_test_correction(test_results)

        # Determine overall significance based on corrected p-values
        significant_results = [
            name
            for name, result in test_results.items()
            if result["corrected_p_value"] < self.config.significance_level
        ]

        # Strategy is considered statistically significant if any test is significant
        passed = len(significant_results) > 0

        # Use the minimum p-value as the score (lower is better)
        min_p_value = min(
            [result["corrected_p_value"] for result in test_results.values()]
        )

        # Create detailed explanation
        explanation = self._create_explanation(
            passed, test_results, significant_results
        )

        return ValidationResult(
            name=self.name,
            passed=passed,
            score=min_p_value,
            threshold=self.config.significance_level,
            explanation=explanation,
        )

    def _run_statistical_tests(
        self, returns: pd.Series, strategy: StrategyInterface
    ) -> Dict[str, Dict]:
        """
        Run all configured statistical tests on the return series.

        Args:
            returns: Series of returns to test
            strategy: The strategy being tested

        Returns:
            Dictionary mapping test names to test results
        """
        test_results = {}

        # Run each requested test
        for test_name in self.config.tests:
            if test_name == "t_test":
                result = self._run_t_test(returns)
            elif test_name == "bootstrap":
                result = self._run_bootstrap_test(returns)
            elif test_name == "wilcoxon":
                result = self._run_wilcoxon_test(returns)
            elif test_name == "jarque_bera":
                result = self._run_jarque_bera_test(returns)
            elif test_name == "shapiro":
                result = self._run_shapiro_test(returns)
            elif test_name == "runs_test":
                result = self._run_runs_test(returns)
            elif test_name == "white_reality_check":
                result = self._run_white_reality_check(returns, strategy)

            test_results[test_name] = result

        return test_results

    def _run_t_test(self, returns: pd.Series) -> Dict:
        """
        Run a t-test to determine if returns are significantly different from null hypothesis.

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "One-Sample t-Test",
            "description": "Tests if returns have a mean significantly different from zero",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Different nulls require different t-test configurations
        if self.config.null_hypothesis == "zero_mean":
            # Test if mean return is significantly different from zero
            t_stat, p_value = stats.ttest_1samp(returns, 0)

            # For one-tailed test (returns > 0), adjust p-value if t-stat is positive
            if (
                "one_tailed" in self.config.custom_parameters
                and self.config.custom_parameters["one_tailed"]
            ):
                if t_stat > 0:
                    p_value = p_value / 2  # one-tailed p-value for positive t-stat
                else:
                    p_value = 1 - (
                        p_value / 2
                    )  # one-tailed p-value for negative t-stat
                result["description"] += " (one-tailed test)"

            result["test_statistic"] = t_stat
            result["p_value"] = p_value
            result["corrected_p_value"] = (
                p_value  # Will be updated if correction is applied
            )
            result["details"] = {
                "mean_return": returns.mean(),
                "std_return": returns.std(),
                "n": len(returns),
                "degrees_of_freedom": len(returns) - 1,
            }

        # Add power analysis for the t-test
        result["details"]["power"] = self._calculate_t_test_power(
            result["details"]["mean_return"],
            result["details"]["std_return"],
            result["details"]["n"],
        )

        return result

    def _calculate_t_test_power(self, mean: float, std: float, n: int) -> float:
        """
        Calculate statistical power of the t-test, the probability of correctly
        rejecting the null hypothesis when it's false.

        Args:
            mean: Mean of returns
            std: Standard deviation of returns
            n: Sample size

        Returns:
            Power estimate between 0 and 1
        """
        # Calculate effect size (Cohen's d)
        effect_size = abs(mean) / std if std > 0 else 0

        # Calculate non-centrality parameter
        ncp = effect_size * (n**0.5)

        # Calculate critical t-value for given significance level
        critical_t = stats.t.ppf(1 - self.config.significance_level / 2, n - 1)

        # Calculate power (probability of exceeding critical value given effect size)
        power = 1 - stats.nct.cdf(critical_t, n - 1, ncp)

        return float(power)

    def _run_bootstrap_test(self, returns: pd.Series) -> Dict:
        """
        Run a bootstrap test to estimate the probability of observed returns under null.

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        # Set parameters
        n_bootstrap = 10000

        result: Dict = {
            "name": "Bootstrap Test",
            "description": "Non-parametric test using bootstrap resampling",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Calculate observed statistic
        if self.config.null_hypothesis == "zero_mean":
            observed_stat = returns.mean()
            stat_name = "mean"
        else:
            # Default to Sharpe ratio for other null hypotheses
            observed_stat = returns.mean() / (
                returns.std() + 1e-10
            )  # Avoid division by zero
            stat_name = "Sharpe ratio"

        # Create bootstrap distribution under the null hypothesis
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            if self.config.null_hypothesis == "zero_mean":
                # For zero mean, resample centered returns (returns - mean)
                centered_returns = returns - returns.mean()
                bootstrap_sample = np.random.choice(
                    centered_returns, size=len(returns), replace=True
                )
                stat = bootstrap_sample.mean()
            else:
                # For other nulls, simple resample with replacement
                bootstrap_sample = np.random.choice(
                    returns, size=len(returns), replace=True
                )
                stat = bootstrap_sample.mean() / (bootstrap_sample.std() + 1e-10)

            bootstrap_stats.append(stat)

        # Calculate p-value (proportion of bootstrap statistics more extreme than observed)
        if observed_stat > 0:
            p_value = np.mean([stat >= observed_stat for stat in bootstrap_stats])
        else:
            p_value = np.mean([stat <= observed_stat for stat in bootstrap_stats])

        result["test_statistic"] = observed_stat
        result["p_value"] = p_value
        result["corrected_p_value"] = (
            p_value  # Will be updated if correction is applied
        )
        result["details"] = {
            "n_bootstrap": n_bootstrap,
            "statistic_name": stat_name,
            "bootstrap_mean": np.mean(bootstrap_stats),
            "bootstrap_std": np.std(bootstrap_stats),
        }

        return result

    def _run_wilcoxon_test(self, returns: pd.Series) -> Dict:
        """
        Run a Wilcoxon signed-rank test (non-parametric alternative to t-test).

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "Wilcoxon Signed-Rank Test",
            "description": "Non-parametric test for median different from zero",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Run the test
        statistic, p_value = stats.wilcoxon(returns, alternative="two-sided")

        # For one-tailed test (returns > 0), adjust p-value based on median
        if (
            "one_tailed" in self.config.custom_parameters
            and self.config.custom_parameters["one_tailed"]
        ):
            if returns.median() > 0:
                p_value = p_value / 2  # one-tailed p-value for positive median
            else:
                p_value = 1 - (p_value / 2)  # one-tailed p-value for negative median
            result["description"] += " (one-tailed test)"

        result["test_statistic"] = statistic
        result["p_value"] = p_value
        result["corrected_p_value"] = (
            p_value  # Will be updated if correction is applied
        )
        result["details"] = {
            "median_return": returns.median(),
            "positive_returns": np.sum(returns > 0),
            "negative_returns": np.sum(returns < 0),
            "zero_returns": np.sum(returns == 0),
        }

        return result

    def _run_jarque_bera_test(self, returns: pd.Series) -> Dict:
        """
        Run Jarque-Bera test for normality of returns.

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "Jarque-Bera Test",
            "description": "Tests if returns follow a normal distribution",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Run the test
        statistic, p_value = stats.jarque_bera(returns)

        result["test_statistic"] = statistic
        result["p_value"] = p_value
        result["corrected_p_value"] = (
            p_value  # Will be updated if correction is applied
        )
        result["details"] = {
            "skewness": stats.skew(returns),
            "kurtosis": stats.kurtosis(returns),
            "n": len(returns),
        }

        # Note for this test: p-value < alpha means returns are NOT normal
        # We'll invert the p-value interpretation in the explanation

        return result

    def _run_shapiro_test(self, returns: pd.Series) -> Dict:
        """
        Run Shapiro-Wilk test for normality of returns.

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "Shapiro-Wilk Test",
            "description": "Tests if returns follow a normal distribution",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Shapiro-Wilk has a sample size limitation
        if len(returns) <= 5000:
            # Run the test
            statistic, p_value = stats.shapiro(returns)

            result["test_statistic"] = statistic
            result["p_value"] = p_value
            result["corrected_p_value"] = (
                p_value  # Will be updated if correction is applied
            )
            result["details"] = {"n": len(returns)}
        else:
            # For large samples, use Anderson-Darling test instead
            result["name"] = "Anderson-Darling Test"
            result["description"] = (
                "Tests if returns follow a normal distribution (used for large samples)"
            )

            statistic, critical_values, significance_levels = stats.anderson(
                returns, "norm"
            )

            # Find the p-value by comparing against critical values
            for sig_level, critical_value in zip(significance_levels, critical_values):
                if statistic <= critical_value:
                    p_value = sig_level / 100  # Convert from percentage
                    break
            else:
                p_value = min(significance_levels) / 100

            result["test_statistic"] = statistic
            result["p_value"] = p_value
            result["corrected_p_value"] = (
                p_value  # Will be updated if correction is applied
            )
            result["details"] = {
                "n": len(returns),
                "critical_values": critical_values.tolist(),
                "significance_levels": significance_levels.tolist(),
            }

        # Note for this test: p-value < alpha means returns are NOT normal
        # We'll invert the p-value interpretation in the explanation

        return result

    def _run_runs_test(self, returns: pd.Series) -> Dict:
        """
        Run a runs test to check for randomness in sequence of returns.

        The runs test checks if the sequence of positive and negative returns
        is random or exhibits patterns/autocorrelation.

        Args:
            returns: Series of returns to test

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "Runs Test",
            "description": "Tests for randomness in the sequence of returns",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Convert returns to binary sequence (1 for positive, 0 for negative)
        binary_returns = (returns > 0).astype(int)

        # Count runs
        runs = 1
        for i in range(1, len(binary_returns)):
            if binary_returns.iloc[i] != binary_returns.iloc[i - 1]:
                runs += 1

        # Count positive and negative returns
        n1 = sum(binary_returns)
        n2 = len(binary_returns) - n1

        # Calculate expected runs and standard deviation under the null hypothesis
        n = n1 + n2
        expected_runs = 1 + (2 * n1 * n2) / n
        std_runs = ((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))) ** 0.5

        # Calculate Z statistic
        z_stat = (runs - expected_runs) / std_runs if std_runs > 0 else 0

        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        result["test_statistic"] = z_stat
        result["p_value"] = p_value
        result["corrected_p_value"] = (
            p_value  # Will be updated if correction is applied
        )
        result["details"] = {
            "runs": runs,
            "expected_runs": expected_runs,
            "positive_returns": n1,
            "negative_returns": n2,
        }

        # Note for this test: p-value < alpha means returns are NOT random
        # We'll invert the p-value interpretation in the explanation

        return result

    def _run_white_reality_check(
        self, returns: pd.Series, strategy: StrategyInterface
    ) -> Dict:
        """
        Run White's Reality Check to test for data snooping bias.

        This is a simplified version of White's Reality Check that tests
        if the strategy's performance is likely due to data mining bias.

        Args:
            returns: Series of returns to test
            strategy: The strategy being tested

        Returns:
            Dictionary containing test results
        """
        result: Dict = {
            "name": "White's Reality Check",
            "description": "Tests if strategy performance is robust to data snooping",
            "test_statistic": None,
            "p_value": None,
            "corrected_p_value": None,
            "details": {},
        }

        # Number of bootstrap simulations
        n_bootstrap = 1000

        # Calculate Sharpe ratio as the performance metric
        observed_sharpe = returns.mean() / (
            returns.std() + 1e-10
        )  # Avoid division by zero

        # Generate bootstrap samples under the null hypothesis
        bootstrap_sharpes = []

        for _ in range(n_bootstrap):
            # Randomly permute the time series to break any patterns
            # This maintains the distribution but destroys any timing/structure
            permuted_returns = returns.sample(frac=1.0, replace=False).reset_index(
                drop=True
            )
            bootstrap_sharpe = permuted_returns.mean() / (
                permuted_returns.std() + 1e-10
            )
            bootstrap_sharpes.append(bootstrap_sharpe)

        # Calculate p-value (proportion of bootstrap metrics better than observed)
        p_value = np.mean([sharpe >= observed_sharpe for sharpe in bootstrap_sharpes])

        result["test_statistic"] = observed_sharpe
        result["p_value"] = p_value
        result["corrected_p_value"] = (
            p_value  # Will be updated if correction is applied
        )
        result["details"] = {
            "n_bootstrap": n_bootstrap,
            "bootstrap_mean_sharpe": np.mean(bootstrap_sharpes),
            "bootstrap_max_sharpe": np.max(bootstrap_sharpes),
        }

        return result

    def _apply_multiple_test_correction(self, test_results: Dict[str, Dict]) -> None:
        """
        Apply correction for multiple comparisons to avoid inflated Type I error.

        Args:
            test_results: Dictionary of test results to correct

        Returns:
            None (modifies test_results in-place)
        """
        # Extract p-values and test names
        test_names = list(test_results.keys())
        p_values = [test_results[name]["p_value"] for name in test_names]

        # Apply correction
        if self.config.correction_method == "bonferroni":
            corrected_p_values = np.minimum(np.array(p_values) * len(p_values), 1.0)
        else:
            # Use statsmodels for other methods
            reject, corrected_p_values, _, _ = multipletests(
                p_values,
                alpha=self.config.significance_level,
                method=self.config.correction_method,
            )

        # Update test results with corrected p-values
        for i, name in enumerate(test_names):
            test_results[name]["corrected_p_value"] = corrected_p_values[i]

    def _create_explanation(
        self,
        passed: bool,
        test_results: Dict[str, Dict],
        significant_results: List[str],
    ) -> str:
        """
        Create a detailed explanation based on statistical analysis results.

        Args:
            passed: Whether the validation passed
            test_results: Dictionary of test results
            significant_results: List of tests that were significant

        Returns:
            Detailed explanation string
        """
        if self.config.null_hypothesis == "zero_mean":
            hypothesis_desc = "returns are not significantly different from zero"
        elif self.config.null_hypothesis == "random_walk":
            hypothesis_desc = "returns follow a random walk"
        else:  # benchmark
            hypothesis_desc = f"returns do not significantly differ from benchmark ({self.config.benchmark_symbol})"

        explanation = (
            f"Statistical significance testing with significance level α = {self.config.significance_level}.\n"
            f"Null hypothesis: {hypothesis_desc}.\n\n"
        )

        if passed:
            explanation += (
                f"Validation PASSED - Found statistically significant evidence "
                f"against the null hypothesis in {len(significant_results)} test(s):\n"
            )
            for name in significant_results:
                result = test_results[name]
                explanation += (
                    f"- {result['name']}: p-value = {result['p_value']:.4f} "
                    f"(corrected: {result['corrected_p_value']:.4f})\n"
                )
        else:
            explanation += (
                f"Validation FAILED - Could not find statistically significant evidence "
                f"against the null hypothesis. No tests were significant at α = {self.config.significance_level}.\n"
            )

        # Add details for all tests
        explanation += "\nDetailed Test Results:\n"

        for _name, result in test_results.items():
            explanation += f"\n## {result['name']}\n"
            explanation += f"Description: {result['description']}\n"

            # Special handling of normality tests (Jarque-Bera, Shapiro)
            if "normality" in result["description"].lower():
                explanation += (
                    f"Test statistic = {result['test_statistic']:.4f}, "
                    f"p-value = {result['p_value']:.4f} "
                    f"(corrected: {result['corrected_p_value']:.4f})\n"
                )

                if result["p_value"] < self.config.significance_level:
                    explanation += "Result: Returns are NOT normally distributed.\n"
                else:
                    explanation += "Result: No evidence against normality of returns.\n"

            # Special handling of runs test
            elif "randomness" in result["description"].lower():
                explanation += (
                    f"Test statistic = {result['test_statistic']:.4f}, "
                    f"p-value = {result['p_value']:.4f} "
                    f"(corrected: {result['corrected_p_value']:.4f})\n"
                )

                if result["p_value"] < self.config.significance_level:
                    explanation += "Result: Returns sequence is NOT random, suggesting patterns or autocorrelation.\n"
                else:
                    explanation += "Result: No evidence against randomness in the sequence of returns.\n"

            # Standard handling for other tests
            else:
                explanation += (
                    f"Test statistic = {result['test_statistic']:.4f}, "
                    f"p-value = {result['p_value']:.4f} "
                    f"(corrected: {result['corrected_p_value']:.4f})\n"
                )

                if result["corrected_p_value"] < self.config.significance_level:
                    explanation += f"Result: Significant at α = {self.config.significance_level}.\n"
                else:
                    explanation += f"Result: Not significant at α = {self.config.significance_level}.\n"

            # Add test details
            for key, value in result["details"].items():
                # Format the value based on type
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    formatted_value = (
                        f"{value:.4f}" if abs(value) < 10000 else f"{value:.2e}"
                    )
                else:
                    formatted_value = str(value)

                explanation += f"- {key}: {formatted_value}\n"

        # Add power analysis interpretation if t-test was run
        if "t_test" in test_results and "power" in test_results["t_test"]["details"]:
            power = test_results["t_test"]["details"]["power"]
            explanation += "\nPower Analysis:\n"

            if power < 0.5:
                explanation += (
                    f"The statistical power is low ({power:.2f}). "
                    f"There's a high risk of Type II error (failing to detect a true effect). "
                    f"Consider collecting more data or looking for strategies with larger effect sizes.\n"
                )
            elif power < 0.8:
                explanation += (
                    f"The statistical power is moderate ({power:.2f}). "
                    f"The test has a reasonable chance of detecting true effects, "
                    f"but there's still some risk of Type II error.\n"
                )
            else:
                explanation += (
                    f"The statistical power is high ({power:.2f}). "
                    f"The test has a good chance of detecting true effects if they exist.\n"
                )

        # Add correction method note if multiple tests were run
        if len(test_results) > 1:
            explanation += (
                f"\nNote: p-values were adjusted using the {self.config.correction_method} "
                f"correction for multiple comparisons.\n"
            )

        # Add interpretive conclusion
        explanation += "\nInterpretative Conclusion:\n"
        if passed:
            if len(significant_results) >= len(test_results) / 2:
                explanation += (
                    "Strong statistical evidence suggests the strategy's performance "
                    "cannot be attributed to chance alone. The results indicate a "
                    "genuinely effective trading strategy.\n"
                )
            else:
                explanation += (
                    "There is some statistical evidence that the strategy's performance "
                    "is not due to chance, but results are mixed. Consider further validation "
                    "with out-of-sample data or additional testing methods.\n"
                )
        else:
            explanation += (
                "Statistical tests do not provide sufficient evidence that the strategy's "
                "performance is different from what could be expected by chance. "
                "This suggests the strategy may not have a true edge in the market.\n"
            )

        return explanation


# Register the validation algorithm
register_validation(
    "StatisticalSignificanceValidation",
    StatisticalSignificanceValidation,
    StatisticalSignificanceConfig,
)
