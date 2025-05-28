# DRL Trading Ingest

## Overview
The `drl-trading-ingest` package is a core component of the AI Trading monorepo responsible for handling comprehensive data ingestion for both training and inference modes. It provides a unified interface for importing financial data from multiple sources and establishes real-time data connections for live trading scenarios.

## Features

### Data Sources Support
- **Local CSV Files**: Import historical data from local CSV files
- **Yahoo Finance**: Fetch market data using Yahoo Finance API
- **AWS S3**: Import and export parquet files from AWS S3 buckets
- **External Data Providers**: Integration with platforms like TwelveData and other financial data APIs

### Training Mode Data Ingestion
- **Multi-source Data Import**: Seamlessly switch between different data sources
- **Historical Data Management**: Comprehensive historical data retrieval and storage
- **Data Validation**: Ensure data quality and consistency across sources
- **Format Standardization**: Convert data from various formats into a unified structure

### Inference Mode Data Ingestion
- **Real-time Data Streaming**: Establish fast, reliable connections to data providers
- **Low-latency Updates**: Optimized for high-frequency trading scenarios
- **Connection Management**: Robust handling of connection failures and reconnections
- **Data Buffering**: Intelligent buffering for consistent data flow

### Data Provider APIs
- **Historical Data Import**:
  - Import from scratch when no historical data exists
  - Fill gaps between latest stored data and current available data
  - Incremental updates to minimize API calls and costs
- **Real-time Connections**:
  - WebSocket connections for live market data
  - REST API fallbacks for reliability
  - Configurable update frequencies

## Getting Started

### Prerequisites
- Python 3.12 or higher
- AWS credentials (for S3 integration)
- API keys for external data providers (TwelveData, etc.)

### Installation
1. Navigate to the package directory:
   ```bash
   cd drl-trading-ingest
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```

### Configuration
Create a configuration file with your API credentials and data source preferences:
```json
{
  "data_sources": {
    "aws_s3": {
      "bucket": "your-trading-data-bucket",
      "region": "us-east-1"
    },
    "twelvedata": {
      "api_key": "your-twelvedata-api-key"
    }
  },
  "default_source": "s3",
  "real_time_provider": "twelvedata"
}
```

### Usage Examples

#### Training Mode - Historical Data Import
```python
from drl_trading_ingest import DataIngestor

# Initialize ingestor
ingestor = DataIngestor(config_path="config.json")

# Import historical data
data = ingestor.import_historical(
    symbol="EURUSD",
    timeframe="H1",
    start_date="2020-01-01",
    end_date="2024-01-01",
    source="s3"  # or "yahoo", "csv", "twelvedata"
)
```

#### Inference Mode - Real-time Data
```python
from drl_trading_ingest import RealTimeIngestor

# Initialize real-time ingestor
rt_ingestor = RealTimeIngestor(config_path="config.json")

# Start real-time data stream
rt_ingestor.start_stream(
    symbols=["EURUSD", "GBPUSD"],
    timeframe="M1",
    callback=your_data_handler
)
```

## Architecture

### Core Components
- **DataIngestor**: Main class for historical data import and management
- **RealTimeIngestor**: Handles real-time data streaming and WebSocket connections
- **DataProviderAPI**: Abstract interface for external data provider integrations
- **StorageManager**: Manages data persistence across different storage backends
- **DataValidator**: Ensures data quality and consistency

### Supported Providers
- **TwelveData**: Professional financial data provider with real-time and historical APIs
- **Yahoo Finance**: Free market data for basic use cases
- **AWS S3**: Cloud storage for large-scale historical data archives
- **Local Storage**: CSV and parquet file support for offline development

## Contributing
Contributions are welcome! Please follow the [contribution guidelines](../CONTRIBUTING.md) to submit issues or pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](../LICENSE.txt) file for details.

## Contact
For questions or support, please contact the development team.
