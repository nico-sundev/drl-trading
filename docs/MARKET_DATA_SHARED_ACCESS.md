# Market Data Shared Access Architecture

## Overview

This document describes the implementation of shared market data access across microservices using SQLAlchemy ORM while maintaining hexagonal architecture principles.

## Architecture Components

### 1. **Entity Layer** (`drl-trading-adapter`)
```
drl-trading-adapter/src/drl_trading_adapter/adapter/database/entity/
└── market_data.py - SQLAlchemy entity mapping to TimescaleDB table
```

**Features:**
- Maps to existing `market_data` table schema
- Maintains TimescaleDB indexes for performance
- Provides ORM-based CRUD operations
- Type-safe data access

### 2. **Port Layer** (`drl-trading-core`)
```
drl-trading-core/src/drl_trading_core/core/port/
└── market_data_reader_port.py - Read-only interface for shared access
```

**Features:**
- Read-only contract for market data access
- Bulk data retrieval for preprocessing
- Latest price access for inference
- Data availability discovery

### 3. **Adapter Layer** (`drl-trading-adapter`)
```
drl-trading-adapter/src/drl_trading_adapter/adapter/database/
├── session_factory.py - SQLAlchemy session management
└── repository/
    └── market_data_repository.py - Entity Framework-style repository
```

**Features:**
- SQLAlchemy ORM-based implementation
- Integrates with existing PostgreSQL connection service
- Read and write operations
- Efficient bulk queries

## Responsibility Segregation

### Write Operations (Ingest Service Only)
- **Owner**: `drl-trading-ingest`
- **Scope**: Schema migrations, data ingestion, write operations
- **Implementation**: Uses `MarketDataRepository` for write operations

### Read Operations (Shared Access)
- **Consumers**: `drl-trading-preprocess`, `drl-trading-inference`, etc.
- **Scope**: Bulk data retrieval, latest prices, data discovery
- **Implementation**: Uses `MarketDataReaderPort` interface

## Integration Pattern

### Service Dependencies
```python
# In preprocessing service
@inject
class MarketDataPreprocessingService:
    def __init__(self, market_data_reader: MarketDataReaderPort):
        self.market_data_reader = market_data_reader
```

### Dependency Injection Configuration
```python
# In service bootstrap
def get_dependency_modules(self, app_config: PreprocessConfig) -> List[Module]:
    return [
        PreprocessModule(app_config),
        CoreModule(),           # Provides ports
        AdapterModule()         # Provides implementations
    ]
```

## Database Configuration

### Shared Configuration Pattern
```yaml
# drl-trading-preprocess/config/application.yaml
infrastructure:
  database:  # Market data database access
    provider: "postgresql"
    host: "${MARKET_DATA_DB_HOST}"
    port: ${MARKET_DATA_DB_PORT:5432}
    database: "marketdata"
    username: "${MARKET_DATA_DB_USER}"
    password: "${MARKET_DATA_DB_PASSWORD}"
```

### Session Management
- **SQLAlchemy Engine**: Managed by `SQLAlchemySessionFactory`
- **Connection Pooling**: Uses NullPool to avoid conflicts with psycopg2 pooling
- **Transaction Management**: Automatic commit/rollback via context managers

## Usage Examples

### Bulk Data for Training
```python
# Get historical data for multiple symbols
training_data = market_data_reader.get_multiple_symbols_data_range(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    timeframe="1H",
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)
```

### Real-time Data for Inference
```python
# Get latest prices for live trading
latest_prices = market_data_reader.get_latest_prices(
    symbols=["EURUSD", "GBPUSD"],
    timeframe="1H"
)
```

### Data Discovery
```python
# Check what data is available
availability = market_data_reader.get_data_availability_summary()
```

## Performance Optimizations

### TimescaleDB Integration
- **Hypertable Partitioning**: Automatic time-based partitioning
- **Composite Indexes**: Optimized for symbol+timeframe+time queries
- **Covering Indexes**: Avoid table lookups for OHLC queries

### SQLAlchemy Query Optimization
- **Bulk Operations**: `session.merge()` for UPSERT behavior
- **Window Functions**: Efficient latest record retrieval
- **IN Clauses**: Optimized multi-symbol queries

## Trade-offs and Benefits

### ✅ Benefits
1. **Performance**: Direct database access, no API overhead
2. **Consistency**: Single source of truth for market data
3. **Entity Framework Style**: Type-safe, maintainable data access
4. **Controlled Evolution**: Ingest service maintains schema ownership
5. **Flexible Queries**: SQLAlchemy enables complex analytical queries

### ⚠️ Trade-offs
1. **Database Coupling**: Services share database dependency
2. **Schema Coordination**: Changes require coordination between services
3. **Testing Complexity**: Integration tests need database setup

## Migration Path

### From Current Implementation
1. **Services continue using existing patterns** during transition
2. **Gradual migration** service by service
3. **Ingest service** gets both interfaces during transition
4. **Deprecate raw SQL** once all services migrated

### Future Enhancements
1. **Read Replicas**: Separate read workloads from write workloads
2. **Caching Layer**: Redis caching for frequently accessed data
3. **Query Analytics**: Monitor and optimize query patterns
4. **Schema Versioning**: Formal schema evolution process

## Conclusion

This architecture provides a pragmatic solution for shared market data access that:
- Balances microservice principles with performance requirements
- Maintains clear ownership and responsibility boundaries
- Provides type-safe, maintainable data access
- Scales well with large time-series datasets
- Supports both batch processing and real-time use cases
