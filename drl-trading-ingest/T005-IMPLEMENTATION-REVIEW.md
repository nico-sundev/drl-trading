# T005 Logging Standardization Implementation - drl-trading-ingest

## Overview

This document describes the implementation of T005 logging standardization in the `drl-trading-ingest` service. This serves as the prototype implementation that will be reviewed before rolling out to other services.

## Implementation Summary

### 🎯 Key Components Implemented

1. **TradingContext DTOs** (`drl-trading-common/model/trading_context.py`)
   - `TradingContext`: Context tracking across services
   - `TradingEventPayload`: Standardized Kafka message structure
   - Progressive context enrichment as data flows through services

2. **ServiceLogger Framework** (`drl-trading-common/logging/`)
   - `TradingLogContext`: Thread-local context management
   - `TradingStructuredFormatter`: JSON formatter for production
   - `TradingHumanReadableFormatter`: Human-readable formatter for development
   - `ServiceLogger`: Main configuration and context management class

3. **Enhanced Configuration**
   - `EnhancedLoggingConfig`: Extended configuration schema
   - Integration with existing `IngestConfig`

4. **Service Integration**
   - Updated `IngestServiceBootstrap` to use T005 logging
   - Enhanced `IngestionService` with trading context tracking
   - Context-aware logging throughout the ingestion pipeline

### 🔧 Features Implemented

#### **Environment-Aware Logging**
- **Development**: Human-readable logs with context information
- **Production/Staging**: JSON structured logs for aggregation systems
- **Automatic Detection**: Based on `DEPLOYMENT_MODE` environment variable

#### **Trading Context Tracking**
- **Correlation ID**: Tracks business operations across services
- **Event ID**: Unique identifier for individual messages/events
- **Symbol**: Financial instrument being processed
- **Progressive Enrichment**: Context grows as data flows through services

#### **Production-Ready Features**
- **Log Rotation**: Configurable file size and backup count
- **File Management**: Separate error logs for production
- **Context Managers**: Easy-to-use patterns for context tracking
- **Performance Optimization**: Structured logging only when needed

#### **OpenTelemetry Preparation**
- **Compatible Structure**: JSON format ready for OTel export
- **Trace Mapping**: correlation_id maps to future trace_id
- **Context Propagation**: Aligns with OTel span context patterns

## 📋 Usage Examples

### Basic Service Configuration

```python
# In service bootstrap
from drl_trading_common.logging.service_logger import ServiceLogger

service_logger = ServiceLogger("drl-trading-ingest", config=self.config.logging.dict())
service_logger.configure()
```

### Context-Aware Logging

```python
# Create trading context
trading_context = TradingContext.create_initial_context(
    symbol="BTCUSDT",
    timeframe="1h"
)

# Use context manager for automatic context tracking
with service_logger.trading_context(trading_context):
    logger.info("Processing market data")  # Automatically includes context
    # ... business logic
    logger.info("Processing completed")     # Context included in all logs
# Context automatically cleared
```

### Market Data Processing

```python
# Specialized context manager for market data
with service_logger.market_data_context("BTCUSDT"):
    logger.info("Starting market data ingestion")
    # Process data...
    logger.info("Market data ingestion completed", extra={
        'records_processed': len(df),
        'operation': 'batch_ingest'
    })
```

## 📊 Log Output Examples

### Development Environment (Human-Readable)
```
2025-08-09 14:30:00 | INFO     | drl_trading_ingest.core.service | drl-trading-ingest | Starting batch market data ingestion [correlation_id=trade-abc-123, symbol=BTCUSDT]
2025-08-09 14:30:01 | INFO     | drl_trading_ingest.core.service | drl-trading-ingest | Market data saved to database [correlation_id=trade-abc-123, symbol=BTCUSDT]
```

### Production Environment (Structured JSON)
```json
{
  "timestamp": "2025-08-09T14:30:00.123Z",
  "service": "drl-trading-ingest",
  "environment": "production",
  "level": "INFO",
  "message": "Starting batch market data ingestion",
  "correlation_id": "trade-abc-123",
  "event_id": "ingest-def-456",
  "symbol": "BTCUSDT",
  "extra_filename": "btcusdt_1h_data.csv",
  "extra_operation": "batch_ingest",
  "trace_id": "trade-abc-123"
}
```

## 🧪 Testing

Comprehensive test suite implemented in `tests/test_t005_logging_standardization.py`:

- ✅ ServiceLogger configuration and setup
- ✅ Human-readable logging in development
- ✅ Structured JSON logging in production
- ✅ Trading context creation and enrichment
- ✅ Context manager functionality
- ✅ Thread-local context management

## 📁 Files Changed/Added

### New Files in drl-trading-common:
- `src/drl_trading_common/model/trading_context.py` - Trading DTOs
- `src/drl_trading_common/logging/trading_log_context.py` - Context management
- `src/drl_trading_common/logging/trading_formatters.py` - Log formatters
- `src/drl_trading_common/logging/service_logger.py` - Main ServiceLogger class
- `src/drl_trading_common/config/enhanced_logging_config.py` - Enhanced config schema

### Modified Files in drl-trading-ingest:
- `infrastructure/config/ingest_config.py` - Added logging configuration
- `infrastructure/bootstrap/ingest_service_bootstrap.py` - T005 integration
- `core/service/ingestion_service.py` - Context-aware logging

### New Files in drl-trading-ingest:
- `config/application-local-t005-example.yaml` - Example configuration
- `tests/test_t005_logging_standardization.py` - Test suite

## 🎯 Review Focus Areas

### 1. **Architecture Alignment**
- ✅ Follows hexagonal architecture principles
- ✅ Logging stays in infrastructure layer
- ✅ Domain objects (TradingContext) remain clean
- ✅ SOLID principles maintained

### 2. **Context Propagation Design**
- ✅ Progressive enrichment across services
- ✅ Service isolation (each service only needs available context)
- ✅ Thread-local storage for safe concurrent operations
- ✅ Automatic context cleanup

### 3. **Production Readiness**
- ✅ Environment-aware configuration
- ✅ Structured JSON for log aggregation
- ✅ Log rotation and file management
- ✅ Error handling and fallback mechanisms

### 4. **Trading Domain Fit**
- ✅ Symbol, correlation_id, event_id align with trading workflows
- ✅ Context flows logically through ingestion → processing → inference → execution
- ✅ Kafka event payloads include structured context
- ✅ Business operations trackable end-to-end

### 5. **Future Extensibility**
- ✅ OpenTelemetry compatibility prepared
- ✅ Additional context fields easily added
- ✅ New formatters can be plugged in
- ✅ Configuration schema extensible

## 🚀 Next Steps After Review

If approved, the next phases will be:

1. **Roll out to other services** using the same patterns
2. **Enhance Kafka message integration** across service boundaries
3. **Add performance monitoring** and optimization
4. **Prepare OpenTelemetry integration** when observability epic starts

## 🔍 Questions for Review

1. **Context Fields**: Do the trading context fields (correlation_id, symbol, event_id) align with your workflow expectations?

2. **Configuration**: Is the EnhancedLoggingConfig schema appropriate for your deployment patterns?

3. **Performance**: Are you comfortable with the JSON formatting overhead in production?

4. **Integration**: Does the bootstrap integration pattern work well with your existing service patterns?

5. **Testing**: Are there additional test scenarios you'd like to see covered?

---

**This implementation serves as the foundation for T005 standardization across all DRL trading services. Please review thoroughly as this pattern will be replicated in all other services.**
