# Strategy Development Guide

> **Quick Start**: Copy `drl-trading-strategy-example` and customize features.

## Strategy Module Structure

```
drl-trading-strategy-{name}/
├── src/drl_trading_strategy_{name}/
│   ├── features/           # Feature implementations
│   ├── indicators/         # Technical indicators
│   ├── environments/       # Trading environments
│   └── module/            # DI module
├── config/
│   └── application-*.yaml  # Environment configs
└── tests/
```

## Feature Development

### 1. Feature Implementation
```python
@feature_type(FeatureTypeEnum.RSI)
class RsiFeature(BaseFeature):
    def compute_all(self) -> Optional[DataFrame]:
        # Batch processing for training
        source_df = self._prepare_source_df()
        return self.indicator_service.compute_rsi(source_df, self.config.length)

    def add(self, df: DataFrame) -> None:
        # Real-time updates (coming soon)
        self.indicator_service.add(self.feature_name, df)

    def compute_latest(self) -> Optional[DataFrame]:
        # Latest values (coming soon)
        return self.indicator_service.get_latest(self.feature_name)
```

### 2. Configuration Schema
```python
@dataclass
class RsiConfig(BaseFeatureConfig):
    length: int = 14

    def __post_init__(self):
        self.feature_type = FeatureTypeEnum.RSI
```

### 3. Feature Registration
```python
# Auto-discovery via decorator
registry.register_feature_class(FeatureTypeEnum.RSI, RsiFeature)
registry.register_config_class(FeatureTypeEnum.RSI, RsiConfig)
```

## Trading Environment

```python
class CustomTradingEnv(BaseTradingEnv):
    def __init__(self, df: DataFrame, config: TradingEnvConfig):
        super().__init__(df, config)

    def _calculate_reward(self) -> float:
        # Custom reward logic
        return portfolio_return - benchmark_return

    def _get_observation(self) -> np.ndarray:
        # Feature vector for agent
        return self.features.iloc[self.current_step].values
```

## Strategy Module DI

```python
class CustomStrategyModule(BaseStrategyModule):
    def as_injector_module(self) -> Module:
        class _Internal(Module):
            @provider
            @singleton
            def provide_feature_registry(self) -> FeatureClassRegistryInterface:
                registry = FeatureClassRegistry()
                registry.discover_feature_classes("custom_strategy.features")
                return registry

            @provider
            @singleton
            def provide_env_type(self) -> Type[BaseTradingEnv]:
                return CustomTradingEnv

        return _Internal()
```

## Configuration

```yaml
# config/application-local.yaml
featuresConfig:
  featureDefinitions:
    - name: "rsi"
      enabled: true
      derivatives: [1]
      parameterSets:
        - enabled: true
          length: 14
        - enabled: true
          length: 21
```

## Development Workflow

```bash
# 1. Create strategy from template
cp -r drl-trading-strategy-example drl-trading-strategy-{name}

# 2. Install in development mode
cd drl-trading-strategy-{name}
uv sync --group dev-full

# 3. Implement features and environment
# Edit src/drl_trading_strategy_{name}/features/
# Edit src/drl_training_strategy_{name}/environments/

# 4. Test strategy
uv run python -m drl_trading_strategy_{name}.main

# 5. Run with core training service
cd ../drl-trading-training
# Configure to use your strategy module
uv run python -m drl_trading_training
```

## Best Practices

- **Feature Isolation**: Each feature in separate file
- **Configuration Driven**: Define features in YAML config
- **Dual Processing**: Support both batch and incremental modes
- **Type Safety**: Full type hints and validation
- **Testing**: Unit tests for features, integration tests for environments

---

*See [Learning Guide](LEARNING_JOURNEY.md) for skill development.*

    def compute_latest(self) -> Optional[DataFrame]:
        """Latest value computation for inference."""
        # Implementation for latest value
        pass
```

#### 2. Trading Environment
Define your RL environment and reward structure:

```python
# src/drl_trading_strategy_example/environments/trading_env.py
from drl_trading_core.environments.base_trading_env import BaseTradingEnv

class CustomTradingEnv(BaseTradingEnv):
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        # Custom environment initialization

    def _calculate_reward(self, action: int, observation: np.ndarray) -> float:
        """Define your reward function."""
        # Custom reward logic
        return reward

    def _get_observation(self) -> np.ndarray:
        """Define observation space."""
        # Custom observation logic
        return observation
```

#### 3. Strategy Module
Wire everything together with dependency injection:

```python
# src/drl_trading_strategy_example/module.py
from injector import Module, provider, singleton
from drl_trading_strategy_example.environments.trading_env import CustomTradingEnv

class ExampleStrategyModule(BaseStrategyModule):
    def as_injector_module(self) -> Module:
        class _Internal(Module):
            @provider
            @singleton
            def provide_feature_registry(self) -> FeatureClassRegistryInterface:
                registry = FeatureClassRegistry()
                registry.discover_feature_classes("drl_trading_strategy_example.features")
                return registry

            @provider
            @singleton
            def provide_env_type(self) -> Type[BaseTradingEnv]:
                return CustomTradingEnv

        return _Internal()
```

## Development Workflow

### 1. Strategy Setup
```bash
# Create new strategy from template
./scripts/create_strategy.sh my-strategy

# Install strategy in development mode
cd drl-trading-strategy-my-strategy
uv sync --group dev-full
```

### 2. Feature Development
```bash
# Define features in config/features.yaml
features:
  - name: "my_custom_feature"
    enabled: true
    derivatives: [1, 5, 10]
    parameters:
      - length: 14
      - smoothing: 3

# Implement feature class
# src/features/my_custom_feature.py
```

### 3. Environment Development
```bash
# Define reward function and observation space
# src/environments/my_trading_env.py

# Test environment locally
uv run python -m pytest tests/unit/environments/
```

### 4. Integration Testing
```bash
# Test with training service
cd drl-trading-training
uv run python -m drl_trading_training \
  --strategy-module drl_trading_strategy_my_strategy \
  --config ../drl-trading-strategy-my-strategy/config/application-local.yaml
```

## Configuration Management

### Strategy Configuration
```yaml
# config/application-local.yaml
app_name: "drl-trading-strategy-example"
version: "1.0.0"

# Strategy-specific settings
strategy:
  lookback_window: 100
  action_space_size: 3
  reward_scaling: 1.0

# Feature definitions reference
features_config: "config/features.yaml"
```

### Feature Configuration
```yaml
# config/features.yaml
features:
  - name: "rsi"
    enabled: true
    derivatives: [1]
    parameters:
      - length: 14
      - length: 21

  - name: "moving_average"
    enabled: true
    derivatives: [1, 5]
    parameters:
      - period: 20
      - type: "sma"
```

## Example Strategy Reference

The [drl-trading-strategy-example](../drl-trading-strategy-example/) provides a minimal reference implementation:

### Included Features
- **RSI Feature**: Relative Strength Index with configurable periods
- **Moving Average Feature**: Simple/Exponential moving averages
- **Price Change Feature**: Price momentum indicators

### Included Environment
- **Basic Trading Environment**: Simple buy/hold/sell actions
- **PnL Reward Function**: Returns-based reward calculation
- **Technical Indicators**: OHLCV + feature-based observations

### Usage Patterns
```bash
# Run example strategy training
cd drl-trading-strategy-example
uv run python -m drl_trading_strategy_example.main

# Integration with training service
cd drl-trading-training
uv run python bootstrap.py \
  --strategy drl_trading_strategy_example \
  --config ../drl-trading-strategy-example/config/application-local.yaml
```

## Best Practices

### Feature Engineering
- **Incremental Processing**: Implement both `compute_all()` and `add()`/`compute_latest()`
- **Configuration Driven**: Use YAML configuration for parameters
- **Type Safety**: Provide proper type hints and validation
- **Testing**: Unit test features with synthetic data

### Environment Design
- **Realistic Constraints**: Model actual trading constraints (slippage, fees)
- **Risk Management**: Include risk controls in reward functions
- **State Management**: Properly handle position tracking and portfolio state
- **Observation Space**: Include relevant market context and technical indicators

### Performance Optimization
- **Vectorization**: Use pandas/numpy operations for feature computation
- **Caching**: Leverage Feast feature store for repeated calculations
- **Memory Management**: Handle large datasets efficiently
- **Profiling**: Monitor performance bottlenecks

---

*See [Architecture Guide](ARCHITECTURE.md) for framework integration details.*
