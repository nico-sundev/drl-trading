---
applyTo: '**'
---
## Modern Microservice Architecture

The DRL Trading system follows a modular microservice architecture with the following components:

### Core Services
- **drl_trading_core**: Framework core providing preprocessing pipeline, training engine, and common services
- **drl_trading_common**: Shared messaging infrastructure, data models, and base interfaces
- **drl_trading_strategy**: Pluggable strategy modules (e.g., drl_trading_strategy_example)
- **drl_trading_training**: Training service with integrated strategy modules
- **drl_trading_inference**: Inference service for real-time predictions
- **drl_trading_ingest**: Market data ingestion service
- **drl_trading_execution**: Trade execution service

### Messaging Infrastructure (drl_trading_common)
- **Pluggable Transport Layer**: Supports both in-memory (training) and RabbitMQ (production) messaging
- **Event-Driven Pipeline**: Market data → Features → Signals → Execution
- **RPC Patterns**: Synchronous inference requests and responses
- **Shared Data Models**: Common message types and configuration schemas across all services

## Data Processing Pipeline

### 1. Loading the Data
- Raw data structured as OHLCV timeseries data and locally stored:
    - symbol 1
        - OHLCV dataset timeframe 1
        - OHLCV dataset timeframe ...
        - OHLCV dataset timeframe n
    - symbol n
        - OHLCV datasets ...
- All timeseries data from different timeframes are loaded
- The lowest timeframe's dataset is called the `base dataset`
- All higher timeframe's datasets are called `other datasets`
- **Responsible classes**: `CsvDataImportService`, `DataImportManager`

### 2. Stripping Other Datasets
- To save computing power, unnecessary data from higher timeframes is removed
- The last timestamp in base dataset serves as the threshold timestamp
- All rows of other datasets after this threshold are removed
- **Responsible classes**: `StripService` (formerly TimeframeStripperService)

### 3. Feature Computing
- Computes all features for given timeframes dataset and symbol
- **Feast Feature Store** used as intelligent cache and feature management
- Each feature has subfeatures defined in Feature classes (extending `BaseFeature`):
    - Example: RsiFeature
        - rsi_7 (for length parameter 7)
        - rsi_14 (for length parameter 14)
- Feature store and feature views created automatically when needed
- Features computed on-demand if not found in store
- **Strategy Integration**: Feature definitions come from strategy modules
- **Processing Modes**:
    - **Batch Processing** (Current): `compute_all()` for full dataset processing
    - **Incremental Processing** (Coming Soon): `add()` and `compute_latest()` for real-time updates
- **Responsible classes**: `FeatureAggregator`, `FeastService`, `ContextFeatureService`

### 4. Merging Timeframes
- Each timeframes dataset compared to the base dataset
- Every base dataset record points to the last confirmed candle's feature value
- **No future sight**: Uses only last closed and confirmed higher timeframe candles
- **Responsible classes**: `MergeService`

### 5. Splitting the Final Dataset
- Final training dataset split into three parts:
    - Training dataset
    - Validation dataset
    - Test dataset
- **Responsible classes**: `SplitService`

### 6. Training Execution
- Create environments and instantiate agents
- **Strategy Integration**: Custom environments and agents from strategy modules
- **Responsible classes**: Strategy's `BaseTradingEnv`, `AgentTrainingService`

## Modern Bootstrap Architecture

### CoreEngine with Dependency Injection
The system uses a modern dependency injection approach via the `CoreEngine` class:

```python
# Initialize with strategy module
engine = CoreEngine(strategy_module)

# Run preprocessing pipeline
results = engine.run_batch_preprocessing(config_path="path/to/config.json")

# Access services through DI container
service = engine.get_service(ServiceClass)
```

### Strategy Module Integration
Strategy modules extend `BaseStrategyModule` and provide:
- **Feature Definitions**: Custom feature classes and configurations
- **Technical Indicators**: Strategy-specific technical indicator implementations
- **Trading Environment**: Custom RL environment for the strategy
- **DI Configuration**: Injector module binding strategy-specific implementations

Example from `drl_trading_strategy_example`:
```python
class ExampleStrategyModule(BaseStrategyModule):
    def as_injector_module(self) -> Module:
        # Returns injector module with strategy bindings
```

### Dependency Injection Container
The `CoreModule` provides:
- **Configuration Management**: Loads and validates `ApplicationConfig`
- **Service Providers**: Singletons for core services (preprocessing, feature computation, etc.)
- **Strategy Integration**: Merges strategy-specific providers with core framework
- **Environment Detection**: Adapts to training vs production deployment modes

### Microservice Bootstrap Modules
Different deployment scenarios use specialized bootstrap modules:
- `TrainingBootstrap`: Configures training pipeline with in-memory messaging
- `InferenceBootstrap`: Sets up inference service with message bus integration
- `DataIngestionBootstrap`: Configures market data ingestion pipeline
- `ExecutionBootstrap`: Sets up trade execution service

## Configuration System

### ApplicationConfig Structure
Modern configuration system based on `ApplicationConfig` class with sections:
- **environment_config**: Deployment environment settings (training/production)
- **features_config**: Feature definitions and computation parameters
- **feature_store_config**: Feast feature store configuration
- **local_data_import_config**: Data import and preprocessing settings
- **rl_model_config**: Reinforcement learning model parameters
- **context_feature_config**: Context feature generation settings

### Configuration Loading
- **File-based**: JSON configuration files matching `ApplicationConfig` schema
- **Environment Variables**: `DRL_TRADING_CONFIG_PATH` for config file location
- **Validation**: Automatic validation during bootstrap process
- **Strategy Integration**: Feature definitions parsed and validated with strategy factory

### Multi-Environment Support
Configuration adapts to deployment context:
- **Training Mode**: Local file paths, in-memory messaging, development settings
- **Production Mode**: Distributed messaging, production endpoints, monitoring enabled

## Integration Patterns

### Service Communication
Services communicate through the `drl_trading_common` messaging infrastructure:

#### Training Mode (Single Process)
- **Direct Service Integration**: CoreEngine directly instantiates and manages services
- **Batch Processing**: Uses `compute_all()` for complete dataset processing
- **No Message Bus**: Training service operates independently via CLI
- **Synchronous Processing**: Pipeline stages execute sequentially

#### Production Mode (Distributed)
- **RabbitMQ Messaging**: Reliable message queuing between microservices
- **Incremental Processing**: Uses `add()` and `compute_latest()` for real-time updates
- **Event-Driven Architecture**: Services publish/subscribe to market data events
- **RPC Patterns**: Synchronous inference requests with async execution

### Strategy Module Integration
Strategy modules are pluggable components that customize:

#### Feature Engineering
```python
# Strategy provides feature registry
feature_registry = strategy_module.get_feature_registry()
feature_factory = strategy_module.get_feature_factory()
```

#### Technical Indicators
```python
# Strategy provides indicator implementations
indicator_factory = strategy_module.get_technical_indicator_factory()
```

#### Trading Environment
```python
# Strategy provides custom RL environment
env_type = strategy_module.get_env_type()
```

### Data Flow Architecture

#### Preprocessing Pipeline
1. **Data Import** → `DataImportManager` loads OHLCV data
2. **Strip Service** → Removes unnecessary future data
3. **Feature Computation** → `FeatureAggregator` computes strategy features
4. **Merge Service** → Combines multi-timeframe data
5. **Split Service** → Creates train/validation/test sets

#### Training Pipeline
1. **Environment Creation** → Strategy's `BaseTradingEnv` instantiated
2. **Agent Training** → `AgentTrainingService` manages RL training
3. **Model Persistence** → Trained models saved for inference
4. **CLI Execution** → Standalone training via command line interface

#### Inference Pipeline (Coming Soon)
1. **Market Data Ingestion** → Real-time price data received
2. **Incremental Feature Updates** → `add()` method processes new data
3. **Latest Feature Computation** → `compute_latest()` generates current values
4. **Model Inference** → Trained model generates trading signals
5. **Signal Execution** → Trading signals sent to execution service

### Deployment Architecture

#### Training Service (`drl_trading_training`)
- **Standalone CLI Service**: No message bus integration, runs independently
- **Direct Dependencies**: Only depends on strategy module, core, and common libraries
- **Batch Processing Focus**: Uses `compute_all()` for full dataset preprocessing
- **Training Pipeline**:
  1. Load strategy module and initialize CoreEngine
  2. Run batch preprocessing pipeline
  3. Split datasets for training/validation/test
  4. Train RL agents using strategy's custom environment
  5. Save trained models for inference service
- **CLI Usage**: `python -m drl_trading_training` with config file
- **Isolation**: Completely decoupled from production messaging infrastructure

#### Inference Service (`drl_trading_inference`)
- **Real-time Processing**: Uses `add()` and `compute_latest()` for incremental updates
- **Message Bus Integration**: Connects to market data via RabbitMQ message bus
- **Lightweight Design**: Loads trained models and strategy configuration
- **Processing Pipeline**:
  1. Subscribe to market data events
  2. Incrementally update features using `add()` method
  3. Compute latest feature values with `compute_latest()`
  4. Generate trading signals using trained models
  5. Publish signals to execution service
- **Dual Processing Support**: Same feature classes work for both training and inference

#### Supporting Services
- **Data Ingest**: Market data collection and normalization
- **Execution**: Order management and trade execution
- **Common Library**: Shared messaging and data models

## Example Usage

### Training Service Implementation
```python
# drl_trading_training: Standalone CLI service
from drl_trading_core.core_engine import CoreEngine
from drl_trading_strategy.module.example_strategy_module import ExampleStrategyModule

class TrainingApp:
    def start_training(self, config_path: Optional[str] = None):
        # No message bus - direct strategy integration
        core_engine = CoreEngine(strategy_module=ExampleStrategyModule())

        # Batch processing pipeline
        final_datasets = core_engine.run_batch_preprocessing(config_path)

        # Training without messaging infrastructure
        split_service = core_engine.get_service(SplitService)
        agent_training_service = core_engine.get_service(AgentTrainingService)

        # CLI execution
        trained_agents = agent_training_service.create_env_and_train_agents(split_datasets)

# Usage: python -m drl_trading_training
```

### Inference Service Implementation (Coming Soon)
```python
# drl_trading_inference: Real-time processing with message bus
from drl_trading_common.messaging import TradingMessageBusFactory, DeploymentMode

class InferenceApp:
    def __init__(self):
        # Production messaging infrastructure
        self.message_bus = TradingMessageBusFactory.create_message_bus(DeploymentMode.PRODUCTION)
        self.core_engine = CoreEngine(strategy_module=ExampleStrategyModule())

    def start_inference(self):
        # Subscribe to market data
        self.message_bus.subscribe_to_market_data("EURUSD", "H1", self.process_market_data)

    def process_market_data(self, market_data):
        # Incremental processing (COMING SOON)
        # feature.add(market_data)  # Add new data point
        # latest_values = feature.compute_latest()  # Get latest features

        # Generate trading signal
        signal = self.trained_model.predict(latest_values)

        # Publish signal
        self.message_bus.publish_trading_signal("EURUSD", signal)
```

### Strategy Module Development
```python
# Enhanced strategy module with decorator-based features
class CustomStrategyModule(BaseStrategyModule):
    def as_injector_module(self) -> Module:
        class _Internal(Module):
            @provider
            @singleton
            def provide_feature_registry(self) -> FeatureClassRegistryInterface:
                registry = FeatureClassRegistry()
                # Automatic discovery of @feature_type decorated classes
                registry.discover_feature_classes("custom_strategy.features")
                return registry

            @provider
            @singleton
            def provide_config_registry(self) -> FeatureConfigRegistryInterface:
                registry = FeatureConfigRegistry()
                # Automatic discovery of configuration classes
                registry.discover_config_classes("custom_strategy.features")
                return registry

            @provider
            @singleton
            def provide_indicator_factory(self) -> TechnicalIndicatorFactoryInterface:
                # Pluggable indicator backend
                return TaLippIndicatorFactory()  # or PandasTaFactory()

            @provider
            @singleton
            def provide_env_type(self) -> Type[BaseTradingEnv]:
                return CustomTradingEnv

        return _Internal()

# Feature implementation with dual processing
@feature_type(FeatureTypeEnum.CUSTOM_SIGNAL)
class CustomSignalFeature(BaseFeature):
    def compute_all(self) -> Optional[DataFrame]:
        # Batch processing for training
        source_df = self._prepare_source_df()
        # ... complete computation logic
        return result_df

    def add(self, df: DataFrame) -> None:
        # Incremental addition (COMING SOON)
        self.indicator_service.add(self.feature_name, df)

    def compute_latest(self) -> Optional[DataFrame]:
        # Latest value computation (COMING SOON)
        return self.indicator_service.get_latest(self.feature_name)
```

## Modern Deployment Patterns

### Docker-based Training
```dockerfile
# Multi-stage build for training
FROM python:3.11-slim as training

# Install framework and strategy
WORKDIR /app
COPY drl-trading-core/ ./framework/
COPY drl-trading-strategy-example/ ./strategy/
RUN pip install -e ./framework/ && pip install -e ./strategy/

# Set training environment
ENV DEPLOYMENT_MODE=training
ENV DRL_TRADING_CONFIG_PATH=/app/strategy/config/applicationConfig.json

CMD ["python", "-m", "drl_trading_training"]
```

### Microservice Production Deployment
```yaml
# docker-compose.production.yml
version: '3.8'
services:
  rabbitmq:
    image: rabbitmq:3-management

  data-ingest:
    build:
      context: .
      dockerfile: Dockerfile.ingest
    environment:
      - DEPLOYMENT_MODE=production
      - RABBITMQ_URL=amqp://rabbitmq:5672

  inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    environment:
      - DEPLOYMENT_MODE=production
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on: [rabbitmq, data-ingest]

  execution:
    build:
      context: .
      dockerfile: Dockerfile.execution
    environment:
      - DEPLOYMENT_MODE=production
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on: [rabbitmq, inference]
```

### CLI Training Deployment
```bash
# Standalone training execution
export DEPLOYMENT_MODE=training
export DRL_TRADING_CONFIG_PATH=./config/applicationConfig.json

# Install dependencies
pip install -e ./drl-trading-core/
pip install -e ./drl-trading-strategy-example/

# Run training (no message bus required)
python -m drl_trading_training
```

## Key Benefits of Modern Architecture

1. **Separation of Concerns**: Core framework, strategies, and deployment logic are clearly separated
2. **Strategy Isolation**: Frequent strategy iterations don't affect core stability
3. **Dual Processing Support**: Same features work for both batch training and real-time inference
4. **Pluggable Indicators**: Easy to upgrade technical indicator libraries without touching features
5. **Configuration-Driven**: Features automatically discovered from JSON configuration
6. **Decorator-Based Mapping**: Clean feature type association without boilerplate code
7. **CLI Training**: Standalone training execution without messaging complexity
8. **Dependency Injection**: Clean service resolution and testing support
9. **Microservice Ready**: Production services use message bus, training runs independently

## Strategy Development Architecture

### Feature-First Strategy Design
Strategy development follows a feature-centric approach with clear separation of concerns:

#### Feature Implementation Pattern
```python
# Modern feature with dual processing support
@feature_type(FeatureTypeEnum.RSI)
class RsiFeature(BaseFeature):
    def compute_all(self) -> Optional[DataFrame]:
        # Batch processing for training/backtesting
        source_df = self._prepare_source_df()
        # ... computation logic

    def add(self, df: DataFrame) -> None:
        # Incremental data addition for real-time
        self.indicator_service.add(self.feature_name, df)

    def compute_latest(self) -> Optional[DataFrame]:
        # Real-time latest value computation
        return self.indicator_service.get_latest(self.feature_name)
```

#### Configuration-Driven Feature Discovery
Features are automatically discovered from configuration using decorators:

```json
// applicationConfig.json
{
  "featuresConfig": {
    "featureDefinitions": [
      {
        "name": "rsi",           // Maps to @feature_type(FeatureTypeEnum.RSI)
        "enabled": true,
        "derivatives": [1],
        "parameterSets": [
          {
            "enabled": true,
            "length": 7          // Passed to RsiConfig
          }
        ]
      }
    ]
  }
}
```

### Current State and Roadmap

### Current Implementation (Batch Processing)
The preprocessing pipeline currently supports only batch processing:
- **Training Focus**: Optimized for complete dataset processing
- **Batch Methods**: All features implement `compute_all()` for full computation
- **Static Data**: Processes historical datasets for model training
- **CLI Training**: `drl_trading_training` runs standalone without message bus

### Coming Soon: Incremental Processing Support
Major enhancement to support real-time inference:

#### Enhanced Feature Interface
```python
class BaseFeature(ABC):
    @abstractmethod
    def compute_all(self) -> Optional[DataFrame]:
        """Batch processing for training/backtesting"""
        pass

    @abstractmethod
    def add(self, df: DataFrame) -> None:
        """Add new data incrementally (COMING SOON)"""
        pass

    @abstractmethod
    def compute_latest(self) -> Optional[DataFrame]:
        """Compute latest values for real-time (COMING SOON)"""
        pass
```

#### Dual-Mode Processing
Features will support both processing modes seamlessly:
- **Training Mode**: Uses `compute_all()` for complete dataset processing
- **Inference Mode**: Uses `add()` + `compute_latest()` for real-time updates
- **Same Code**: Identical feature implementations work in both contexts

#### Real-Time Pipeline Benefits
- **Low Latency**: Incremental updates avoid recomputing entire datasets
- **Memory Efficient**: Only latest values maintained in production
- **Scalable**: Supports high-frequency market data streams
- **Consistent**: Same feature logic ensures training/inference parity

### Architectural Evolution Timeline

#### Phase 1: Current (Batch Only)
- ✅ Complete batch preprocessing pipeline
- ✅ Strategy module integration
- ✅ CLI training service
- ✅ Feature discovery and configuration system

#### Phase 2: Incremental Support (In Development)
- 🔄 Enhanced `BaseFeature` interface with `add()` and `compute_latest()`
- 🔄 Indicator service streaming support
- 🔄 Real-time feature computation
- 🔄 Inference service implementation

#### Phase 3: Production Ready (Future)
- ⏳ Complete microservice messaging integration
- ⏳ High-frequency data processing
- ⏳ Production monitoring and observability
- ⏳ Auto-scaling and fault tolerance
