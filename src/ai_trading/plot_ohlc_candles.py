from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore
import pandas as pd
from pandas import DataFrame


class OHLCPlotter:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the OHLCPlotter with an OHLC DataFrame.
        :param df: DataFrame with columns ["Close", "High", "Low"].
        """
        self.df = df.copy()
        self._validate_data()

    def _validate_data(self):
        """Ensure DataFrame has required columns and generate synthetic index for plotting."""
        required_columns = {"Close", "High", "Low"}
        if not required_columns.issubset(self.df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Generate Open prices as midpoint (for demonstration)
        # self.df["Open"] = (self.df["Close"] + self.df["Low"]) / 2

        # Add a dummy index to make it time-series-like
        self.df["Date"] = pd.date_range(
            start="2024-01-01", periods=len(self.df), freq="D"
        )
        self.df.set_index("Date", inplace=True)

    def plot(self, title="OHLC Candlestick Chart"):
        """
        Plot the OHLC data as a candlestick chart.
        :param title: Title of the chart.
        """
        mpf.plot(
            self.df,
            type="candle",
            style="charles",
            title=title,
            ylabel="Price",
            volume=False,
        )
        plt.show()


def plot_candlesticks(
    df: DataFrame, analysis_results: Optional[Dict[str, Any]] = None
) -> None:
    """Plot candlesticks with optional analysis results overlaid"""
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["Time"])

    # Create candlestick plot
    fig, axlist = mpf.plot(
        df,
        type="candle",
        title="Price Chart with Analysis",
        volume=True,
        style="yahoo",
        figsize=(15, 10),
        panel_ratios=(3, 1),
        returnfig=True,
    )

    # Plot analysis results if provided
    if analysis_results:
        _plot_analysis_results(df, analysis_results, axlist[0])

    plt.show()


def _plot_analysis_results(
    df: DataFrame, results: Dict[str, Any], ax: plt.Axes
) -> None:
    """Plot additional analysis results on the candlestick chart"""
    if "trades" in results:
        _plot_trade_markers(df, results["trades"], ax)
    if "support_levels" in results:
        _plot_horizontal_lines(results["support_levels"], ax, "g--", "Support")
    if "resistance_levels" in results:
        _plot_horizontal_lines(results["resistance_levels"], ax, "r--", "Resistance")


def _plot_trade_markers(
    df: DataFrame, trades: List[Dict[str, Any]], ax: plt.Axes
) -> None:
    """Plot trade entry and exit points"""
    for trade in trades:
        entry_idx = df.index[trade["entry_step"]]
        exit_idx = df.index[trade["exit_step"]]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]

        # Plot entry point
        color = "g" if trade["direction"] == "long" else "r"
        ax.plot(entry_idx, entry_price, f"{color}^", markersize=10, label="Entry")

        # Plot exit point
        ax.plot(exit_idx, exit_price, "kv", markersize=10, label="Exit")


def _plot_horizontal_lines(
    levels: List[float], ax: plt.Axes, style: str, label: str
) -> None:
    """Plot horizontal lines for support/resistance levels"""
    for level in levels:
        ax.axhline(y=level, color=style[0], linestyle=style[1:], alpha=0.5, label=label)
