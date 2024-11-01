import os
import pandas as pd


class LocalDataImportService:

    def __init__(self):
        pass

    # Download CSV data from a given URL
    def _download_data(self, url):
        return pd.read_csv(
            url,
            usecols=["Time", "Open", "High", "Low", "Close"],
            # index_col="Time",
            sep="\t",
            skipinitialspace=True,  # Remove leading/trailing whitespace in headers
            parse_dates=["Time"],
        )

    # Read local data source and store into dataframe
    def import_data(self, limit) -> dict:
        # Example usage:
        urls = {
            "H1": os.path.join(
                os.path.dirname(__file__), "..\\..\\..\\data\\raw\\EURUSD_H1.csv"
            ),
            "H4": os.path.join(
                os.path.dirname(__file__), "..\\..\\..\\data\\raw\\EURUSD_H4.csv"
            ),
            "D1": os.path.join(
                os.path.dirname(__file__), "..\\..\\..\\data\\raw\\EURUSD_D1.csv"
            ),
        }

        # Download 1H, 4H and 1D datasets from the URLs
        return {
            "H1": self._download_data(urls["H1"]).head(limit),
            "H4": self._download_data(urls["H4"]).head(limit),
            "D1": self._download_data(urls["D1"]).head(limit),
        }
