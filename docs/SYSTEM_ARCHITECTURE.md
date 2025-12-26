# System Architecture

This document provides detailed architectural views of the DRL Trading Framework, including workflow diagrams, service communication patterns, and key design decisions.

## Architecture Principles

The framework follows these core principles:

- **Hexagonal Architecture**: Clean separation between business logic (core), external interfaces (adapters), and application concerns
- **Event-Driven Communication**: Services communicate asynchronously via Kafka topics for loose coupling
- **Separation of Concerns**: Strategy logic is decoupled from framework infrastructure
- **Testability**: Dependency injection and ports/adapters enable comprehensive testing
- **Scalability**: Microservices can scale independently based on load

## System Workflows

The DRL Trading Framework operates through two distinct workflows:

## Offline Training Workflow

Batch processing for model training using historical data:

```mermaid
sequenceDiagram
    participant Training
    participant Ingest
    participant DP as Data Provider
    participant PG as PostgreSQL<br/>(TimescaleDB)
    participant Kafka
    participant Preprocess
    participant Feast as Feast Offline<br/>(Parquet)
    participant MLflow

    Note over Training,MLflow: Offline Training Workflow<br/>(User initiates training with custom strategy implementation)

    Training->>Ingest: Submit a training data request<br/>(symbol, timeframe, date range)
    Ingest-->>Training: Return confirmation
    Ingest->>DP: Fetch historical OHLCV
    DP-->>Ingest: 1m OHLCV data

    Ingest->>PG: Write raw 1m data
    PG-->>Ingest: Confirm write

    Ingest-)Kafka: Publish: requested.preprocess-data

    Kafka--)Preprocess: Consume: requested.preprocess-data
    Preprocess->>PG: Read 1m OHLCV data
    PG-->>Preprocess: Return raw data

    Preprocess->>Preprocess: Resample 1m → Higher TFs
    Preprocess->>PG: Write HTF OHLCV data
    PG-->>Preprocess: Confirm write

    Preprocess->>Preprocess: Compute features<br/>(technical indicators)
    Preprocess->>Feast: Write feature vectors<br/>(batch materialization)
    Feast-->>Preprocess: Confirm storage

    Preprocess-)Kafka: Publish: completed.preprocess-data.offline

    Kafka--)Training: Consume: completed.preprocess-data.offline

    Training->>Feast: Load training dataset<br/>(historical features)
    Feast-->>Training: Return feature matrix

    Training->>Training: Train RL model<br/>(Stable-B3 + custom reward)

    Training->>MLflow: Register model artifacts<br/>(weights, metadata, metrics)
    MLflow-->>Training: Model version ID

    Training->>Training: Backtest the trained model
```

## Online Trading Workflow

Real-time processing for live trading:

```mermaid
sequenceDiagram
    participant DP as Data Provider
    participant Ingest
    participant PG as PostgreSQL<br/>(TimescaleDB)
    participant Kafka
    participant Preprocess
    participant Redis as Feast Online<br/>(Redis)
    participant Inference
    participant MLflow
    participant Execution
    participant Broker

    Note over DP,Broker: Online Trading Workflow (Real-time Processing)

    Ingest->>DP: Request historical 1m OHLCV data (fill gaps)
    DP-->>Ingest: Return missing 1m OHLCV data
    Ingest->>PG: Write historical 1m data
    PG-->>Ingest: Confirm write
    Ingest-)Kafka: Publish: requested.preprocess-data (Feature Warmup)

    Kafka--)Preprocess: Consume: requested.preprocess-data
    Preprocess-->>Preprocess: Compute features on historical data
    Preprocess-)Kafka: Publish: completed.preprocess-data.catchup

    Kafka--)Ingest: Consume: completed.preprocess-data.catchup
    Ingest-)DP: Initiate live 1m OHLCV Data Streaming
    DP--)Ingest: Stream live 1m OHLCV
    Ingest->>PG: Write real-time 1m data
    PG-->>Ingest: Confirm write
    Ingest-)Kafka: Publish: requested.preprocess-data

    Kafka--)Preprocess: Consume: requested.preprocess-data
    Preprocess->>PG: Read recent 1m data
    PG-->>Preprocess: Return raw 1m OHLCV data

    Preprocess->>Preprocess: Resample 1m → Higher TFs<br/>(rolling window)
    Preprocess-)Kafka: Publish: requested.store-resampled-data

    Kafka--)Ingest: Consume: requested.store-resampled-data
    Ingest->>PG: Write resampled live data
    PG-->>Ingest: Confirm write

    Preprocess->>Preprocess: Compute features<br/>(real-time indicators)
    Preprocess->>Redis: Write latest features<br/>(online materialization)
    Redis-->>Preprocess: Confirm cache

    Preprocess-)Kafka: Publish: completed.preprocess-data.online

    Kafka--)Inference: Consume: completed.preprocess-data.online

    Inference->>Redis: Get latest features
    Redis-->>Inference: Return feature vector

    Inference->>MLflow: Load latest model
    MLflow-->>Inference: Return model weights

    Inference->>Inference: Generate prediction<br/>(buy/sell/hold)

    Inference-)Kafka: Publish: completed.predict<br/>(action + confidence)

    Kafka--)Execution: Consume: completed.predict

    Execution->>Execution: Validate signal<br/>(risk checks)

    Execution->>Broker: Place order<br/>(market/limit)
    Broker-->>Execution: Order confirmation

    Execution->>PG: Write trade record
    PG-->>Execution: Confirm write

    Note over DP,Broker: Cycle repeats on new data
```

**Key Characteristics:**
- **Streaming data ingestion**: Continuous flow of market data from external providers
- **Online feature serving**: Low-latency feature access via Redis for real-time predictions
- **Event-driven coordination**: Kafka topics coordinate the prediction and execution pipeline
- **Risk management**: Execution service validates signals before placing orders

## Key Design Decisions

### Why Hexagonal Architecture?

**Benefits:**
- **Testability**: Business logic can be tested without external dependencies
- **Flexibility**: Swap implementations (e.g., Kafka → Redis, Binance → Interactive Brokers) without changing core logic
- **Maintainability**: Clear boundaries between layers reduce coupling

**Structure per service:**
```
src/drl_trading_{service}/
├── adapter/           # External interfaces (Kafka, REST, databases)
├── core/
│   ├── port/         # Business contracts (interfaces)
│   └── service/      # Business logic (implementation)
└── application/
    ├── config/       # Configuration classes
    └── di/           # Dependency injection setup
```

### Why Event-Driven with Kafka?

**Benefits:**
- **Loose coupling**: Services don't need to know about each other
- **Scalability**: Add consumers without modifying producers
- **Resilience**: Message persistence ensures no data loss on failures
- **Async processing**: Long-running tasks (training, backtesting) don't block other services

**Topic naming convention:**
- `requested.*` - Service requests (e.g., `requested.preprocess-data`)
- `completed.*` - Task completions (e.g., `completed.preprocess-data.offline`)
- `error.*` - Error events for monitoring
- `dlq.*` - Dead letter queue for failed messages

### Why Separate Feast Offline and Online Stores?

**Offline (Parquet):**
- Optimized for batch access during training
- Columnar storage for efficient feature retrieval
- Supports time-travel (historical point-in-time correctness)

**Online (Redis):**
- Low-latency key-value access for real-time inference
- In-memory for sub-millisecond lookups
- Only stores latest feature values

This separation follows ML best practices: **train on historical data, serve with real-time data**.

### Why Strategy Decoupling?

The strategy module (`drl-trading-strategy`) is intentionally separated:

**Benefits:**
- **IP protection**: Keep proprietary trading logic private
- **Modularity**: Swap strategies without framework changes
- **Testing**: Framework can be tested with example strategy

**Integration point**: Strategy provides:
- Custom reward functions for RL training
- Trading signal validation rules
- Risk management parameters

## Infrastructure Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Data Ingestion** | Market data streaming | drl-trading-ingest |
| **Feature Engineering** | Resampling & indicator computation | drl-trading-preprocess |
| **Model Training** | RL model optimization | drl-trading-training + Stable-B3 |
| **Inference** | Real-time predictions | drl-trading-inference |
| **Execution** | Order placement & management | drl-trading-execution |
| **Message Bus** | Event coordination | Kafka |
| **Feature Store** | ML feature management | Feast (Parquet + Redis) |
| **Model Registry** | Model versioning | MLflow |
| **Time-Series DB** | OHLCV data storage | PostgreSQL + TimescaleDB |

---

**→ Next Steps:**
- **Implement a service**: [Developer Guide](DEVELOPER_GUIDE.md)
- **Deploy infrastructure**: [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **Create a strategy**: [Strategy Development](STRATEGY_DEVELOPMENT.md)
