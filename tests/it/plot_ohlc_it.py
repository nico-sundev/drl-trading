import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

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
        #self.df["Open"] = (self.df["Close"] + self.df["Low"]) / 2
        
        # Add a dummy index to make it time-series-like
        self.df["Date"] = pd.date_range(start="2024-01-01", periods=len(self.df), freq="D")
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
            volume=False
        )
        plt.show()

# Example Usage:
ohlc_data = pd.DataFrame({
    "Open": [99, 100, 102, 101, 105, 103, 108, 106],
    "Close": [100, 102, 101, 105, 103, 108, 106, 110],
    "High":  [101, 103, 102, 107, 104, 110, 107, 113],
    "Low":   [99, 100, 100, 104, 102, 107, 104, 109]
})

plotter = OHLCPlotter(ohlc_data)
plotter.plot()
