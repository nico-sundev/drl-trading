from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, List, Optional

from drl_trading_core.backtesting.validation.container.overall_status import (
    OverallStatus,
)
from drl_trading_core.backtesting.validation.container.validation_result import (
    ValidationResult,
)


class ValidationResultNotFoundError(Exception):
    """Raised when a validation result is not found by name."""

    pass


@dataclass
class BacktestValidationSummary:
    """
    Summary of validation results from a backtest run.

    This class provides efficient O(1) access to validation results by name while
    maintaining compatibility with existing code that expects lists of results.

    Attributes:
        overall_status: Overall pass/fail status of the validation
        _results_dict: Dictionary of validation results indexed by name (internal)
        failed_algorithms: List of names of failed validation algorithms
        passed_algorithms: List of names of passed validation algorithms
        config_snapshot: Configuration used for the validation
    """

    overall_status: OverallStatus
    _results_dict: Dict[str, ValidationResult] = field(default_factory=dict, repr=False)
    failed_algorithms: List[str] = field(default_factory=list)
    passed_algorithms: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Version of the data structure, useful for migration/compatibility
    _VERSION: ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Ensure the internal data structure is properly initialized."""
        # Handle legacy initialization with list of results
        if hasattr(self, "results") and isinstance(
            getattr(self, "results", None), list
        ):
            # Migrate from old format (list) to new format (dict)
            results_list = self.results
            for result in results_list:
                self._results_dict[result.name] = result
            # Remove the old attribute to avoid confusion
            delattr(self, "results")

    @property
    def results(self) -> List[ValidationResult]:
        """
        Get all validation results as a list.

        Maintains backward compatibility with code expecting a list of results.

        Returns:
            List of ValidationResult objects
        """
        return list(self._results_dict.values())

    @classmethod
    def from_results(
        cls,
        results: List[ValidationResult],
        overall_status: OverallStatus,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> "BacktestValidationSummary":
        """
        Create a summary from a list of validation results.

        Args:
            results: List of validation results
            overall_status: Overall status of the validation
            config_snapshot: Configuration used for the validation

        Returns:
            A new BacktestValidationSummary instance
        """
        # Initialize with empty collections
        summary = cls(
            overall_status=overall_status, config_snapshot=config_snapshot or {}
        )

        # Process each result
        for result in results:
            summary._results_dict[result.name] = result
            if result.passed:
                summary.passed_algorithms.append(result.name)
            else:
                summary.failed_algorithms.append(result.name)

        return summary

    def get_result_by_name(self, name: str) -> Optional[ValidationResult]:
        """
        Retrieve a specific validation result by algorithm name with O(1) lookup.

        Args:
            name: The name of the validation algorithm

        Returns:
            ValidationResult if found, None otherwise

        Examples:
            >>> summary = BacktestValidationSummary(...)
            >>> monte_carlo_result = summary.get_result_by_name("MonteCarloValidation")
            >>> if monte_carlo_result:
            >>>     print(monte_carlo_result.explanation)
        """
        return self._results_dict.get(name)

    def get_result(self, name: str) -> ValidationResult:
        """
        Retrieve a specific validation result by algorithm name with strict error checking.

        Unlike get_result_by_name, this method raises an exception when the result is not found,
        which can be useful for situations where the result is expected to exist.

        Args:
            name: The name of the validation algorithm

        Returns:
            ValidationResult

        Raises:
            ValidationResultNotFoundError: If no result with the given name exists

        Examples:
            >>> summary = BacktestValidationSummary(...)
            >>> try:
            >>>     monte_carlo_result = summary.get_result("MonteCarloValidation")
            >>>     print(monte_carlo_result.explanation)
            >>> except ValidationResultNotFoundError:
            >>>     print("Required validation was not run")
        """
        result = self._results_dict.get(name)
        if result is None:
            raise ValidationResultNotFoundError(
                f"No validation result found with name '{name}'"
            )
        return result

    def add_result(self, result: ValidationResult) -> None:
        """
        Add a new validation result to the summary.

        Args:
            result: The validation result to add

        Note:
            If a result with the same name already exists, it will be overwritten.
            This will also update the passed_algorithms or failed_algorithms lists.
        """
        # Check if we're updating an existing result
        existing = self._results_dict.get(result.name)
        if existing is not None:
            # Remove from appropriate list
            if existing.passed and result.name in self.passed_algorithms:
                self.passed_algorithms.remove(result.name)
            elif not existing.passed and result.name in self.failed_algorithms:
                self.failed_algorithms.remove(result.name)

        # Add the new result
        self._results_dict[result.name] = result

        # Update the appropriate list
        if result.passed:
            self.passed_algorithms.append(result.name)
        else:
            self.failed_algorithms.append(result.name)

    def get_passed_results(self) -> List[ValidationResult]:
        """
        Get all validation results that passed.

        Returns:
            List of passing ValidationResult objects
        """
        return [
            self._results_dict[name]
            for name in self.passed_algorithms
            if name in self._results_dict
        ]

    def get_failed_results(self) -> List[ValidationResult]:
        """
        Get all validation results that failed.

        Returns:
            List of failing ValidationResult objects
        """
        return [
            self._results_dict[name]
            for name in self.failed_algorithms
            if name in self._results_dict
        ]

    def __iter__(self) -> Iterator[ValidationResult]:
        """
        Allow iterating through all validation results.

        Returns:
            Iterator over ValidationResult objects
        """
        return iter(self.results)

    def __len__(self) -> int:
        """
        Get the number of validation results.

        Returns:
            Number of validation results
        """
        return len(self._results_dict)

    def __contains__(self, name: str) -> bool:
        """
        Check if a validation result with the given name exists.

        Args:
            name: The name to check for

        Returns:
            True if a result with the given name exists, False otherwise
        """
        return name in self._results_dict
