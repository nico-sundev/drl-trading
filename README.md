# Feature Store Integration

This directory contains the Feast feature store integration for the AI Trading project. The feature store is used to cache and serve computed features, avoiding redundant computations and enabling feature reuse.

## Structure

- `feature_store_service.py`: Handles creation of feature views based on feature definitions
- `feature_store.yaml`: Feast configuration file (auto-generated on first run)

## Usage

The feature store is configured through the `featureStoreConfig` section in `applicationConfig.json`. By default, it is disabled to avoid overhead during development.

To enable the feature store:

1. Set `"enabled": true` in the `featureStoreConfig` section
2. Features will automatically be stored and retrieved from the feature store
3. Feature views are created dynamically based on your feature definitions

## Feature Views

Each feature (RSI, MACD, etc.) with its specific parameter set gets its own feature view. For example:
- `rsi_[hash]` for RSI with length=14
- `macd_[hash]` for MACD with fast=12, slow=26, signal=9

The hash is generated from the parameter set to ensure unique views for different configurations.

## Storage

Features are stored in parquet files as specified in `offline_store_path`. The files are partitioned by timestamp for efficient retrieval.
