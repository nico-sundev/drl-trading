# DRL Trading Common Library

Shared library containing common components for the DRL Trading System, including messaging infrastructure, data models, and utilities.

## Features

- **Messaging System**: Pluggable transport layer supporting in-memory (training) and RabbitMQ (production) modes
- **Trading Data Models**: Standardized data structures for market data, signals, and execution events
- **Deployment Abstractions**: Environment-based configuration for different deployment scenarios
- **Utility Functions**: Common helpers for all trading services

## Installation

### Basic Installation
```bash
pip install -e .
```

### With RabbitMQ Support (Production)
```bash
pip install -e ".[rabbitmq]"
```

### With All Dependencies (Development)
```bash
pip install -e ".[all]"
```

## Quick Start

### Training Mode (In-Memory)
```python
import os
from drl_trading_common.messaging import TradingMessageBus, DeploymentMode

# Set environment for training
os.environ['DEPLOYMENT_MODE'] = 'training'

# Create message bus
bus = TradingMessageBus(DeploymentMode.TRAINING)
bus.start()

# Publish market data
bus.publish_market_data('EURUSD', 'H1', {
    'open': 1.2340, 'close': 1.2350, 'timestamp': '2024-01-01T10:00:00Z'
})

# Subscribe to signals
def handle_signal(data):
    print(f"Signal: {data}")

bus.subscribe_to_trading_signals('EURUSD', handle_signal)
```

### Production Mode (RabbitMQ)
```python
import os
from drl_trading_common.messaging import TradingMessageBus, DeploymentMode

# Set environment for production
os.environ['DEPLOYMENT_MODE'] = 'production'
os.environ['RABBITMQ_URL'] = 'amqp://guest:guest@localhost:5672/'

# Create message bus with RabbitMQ
bus = TradingMessageBus(DeploymentMode.PRODUCTION)
bus.start()

# Same API, different transport!
bus.publish_market_data('EURUSD', 'H1', market_data)
```

## Architecture

The library provides a **pluggable transport layer** that allows the same code to work in different deployment scenarios:

- **Training Mode**: Fast in-memory communication for single-process training
- **Production Mode**: Reliable RabbitMQ messaging for distributed microservices

## Services Using This Library

- `drl-trading-framework`: Core ML training and inference
- `drl-trading-ingest`: Market data ingestion service
- `drl-trading-execution`: Trade execution service
- `drl-trading-impl-example`: Reference implementation

## Demo

Run the comprehensive messaging demo to see all features in action:

```bash
python demo_messaging_modes.py
```

This demo shows:
- **Training Mode**: Fast in-memory messaging for single-process training
- **Production Simulation**: Multi-threaded services communicating via message bus
- **Environment Switching**: Toggling between deployment modes
- **RPC Patterns**: Synchronous inference requests
- **Event-Driven Pipeline**: Market data → Features → Signals → Execution

## Development

```bash
# Install for development
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python demo_messaging_modes.py

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/
```
