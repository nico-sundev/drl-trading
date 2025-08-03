# T002: ServiceConfigLoader Enhancement or Replacement

**Priority:** High
**Estimated Effort:** 4 days
**Type:** Implementation
**Depends On:** T001 (Configuration Architecture Research)

## Objective
Implement the standardized configuration management solution based on T001 research decisions.

## Scope
Either enhance the existing `ServiceConfigLoader` or implement a replacement using the chosen configuration library with unified patterns for all services.

## Implementation Options

### Option A: Enhance ServiceConfigLoader
If research indicates keeping the custom solution:

#### Required Enhancements:
1. **Schema Validation Integration**
   ```python
   from pydantic import BaseModel

   class ServiceConfigLoader:
       @staticmethod
       def load_config_with_validation(
           config_class: Type[BaseModel],
           schema_class: Type[BaseModel],
           **kwargs
       ) -> BaseModel:
           # Load and validate configuration
   ```

2. **Secret Substitution Support**
   ```python
   # Support for ${VAR} substitution in YAML/JSON
   database:
     url: ${DATABASE_URL}
     password: ${DB_PASSWORD:default_value}
   ```

3. **Enhanced Environment Detection**
   ```python
   @staticmethod
   def detect_environment() -> EnvironmentType:
       # Improved environment detection logic
   ```

### Option B: Replace with Chosen Library
If research indicates adopting a third-party solution:

#### Implementation Tasks:
1. **New Configuration Module**
   ```python
   # drl_trading_common/config/unified_config_loader.py
   from abc import ABC, abstractmethod
   from typing import Type, TypeVar, Optional

   T = TypeVar('T', bound=BaseApplicationConfig)

   class ConfigurationLoader(ABC):
       @abstractmethod
       def load_service_config(
           self,
           config_class: Type[T],
           service_name: str,
           environment: Optional[str] = None
       ) -> T:
           pass

   class PydanticSettingsLoader(ConfigurationLoader):
       # Implementation using Pydantic Settings

   class DynaconfLoader(ConfigurationLoader):
       # Implementation using Dynaconf
   ```

2. **Factory Pattern**
   ```python
   class ConfigLoaderFactory:
       @staticmethod
       def create_loader(loader_type: str = "default") -> ConfigurationLoader:
           # Factory method to create appropriate loader
   ```

## Tasks

### 1. Core Implementation
**Duration:** 2 days

#### For ServiceConfigLoader Enhancement:
- [ ] Add schema validation support using Pydantic
- [ ] Implement secret substitution with `${VAR}` syntax
- [ ] Add comprehensive error handling and validation
- [ ] Enhance environment detection and precedence rules
- [ ] Add configuration caching for performance

#### For New Library Implementation:
- [ ] Create unified configuration loader interface
- [ ] Implement chosen library integration
- [ ] Add factory pattern for different loader types
- [ ] Implement environment-specific configuration loading
- [ ] Add secret management integration

#### Common Requirements:
```python
# Expected interface for all loaders
class BaseConfigLoader:
    def load_config(
        self,
        config_class: Type[T],
        service_name: str,
        config_path: Optional[str] = None,
        environment: Optional[str] = None,
        validate_schema: bool = True
    ) -> T:
        """Load and validate service configuration."""
        pass

    def reload_config(self, config_instance: T) -> T:
        """Reload configuration (useful for development)."""
        pass

    def get_environment(self) -> str:
        """Get current deployment environment."""
        pass
```

### 2. Environment Variable Handling
**Duration:** 1 day

#### Implementation:
```python
# Support for multiple override patterns
class EnvironmentOverride:
    @staticmethod
    def apply_overrides(
        config_dict: dict,
        prefix: str = "",
        delimiter: str = "__"
    ) -> dict:
        """Apply environment variable overrides to configuration."""

    @staticmethod
    def substitute_secrets(config_dict: dict) -> dict:
        """Replace ${VAR} placeholders with environment values."""
```

#### Features:
- [ ] Nested configuration override with `SERVICE__SECTION__KEY`
- [ ] Secret substitution with `${VAR}` and `${VAR:default}`
- [ ] Type coercion for environment variables
- [ ] Environment variable validation and error reporting

### 3. Configuration Schema Validation
**Duration:** 1 day

#### Pydantic Integration:
```python
from pydantic import BaseModel, validator, Field
from typing import Optional, List

class DatabaseConfig(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(5432, ge=1, le=65535)
    username: str
    password: str = Field(..., min_length=8)

    @validator('host')
    def validate_host(cls, v):
        # Custom validation logic
        return v

class ServiceConfig(BaseApplicationConfig):
    database: DatabaseConfig
    features: Optional[List[str]] = []

    class Config:
        env_prefix = "SERVICE_"
        case_sensitive = False
```

#### Schema Features:
- [ ] Automatic validation on configuration load
- [ ] Custom validators for business logic
- [ ] Clear error messages for invalid configurations
- [ ] Environment variable type coercion
- [ ] Default value handling

### 4. Testing & Documentation
**Duration:** Included in implementation

#### Unit Tests:
```python
class TestUnifiedConfigLoader:
    def test_load_basic_config(self):
        """Test basic configuration loading."""

    def test_environment_overrides(self):
        """Test environment variable overrides."""

    def test_secret_substitution(self):
        """Test secret placeholder substitution."""

    def test_schema_validation(self):
        """Test configuration schema validation."""

    def test_environment_detection(self):
        """Test environment detection logic."""
```

#### Integration Tests:
- [ ] Test with real service configurations
- [ ] Test Docker environment integration
- [ ] Test CI/CD pipeline compatibility
- [ ] Test performance with large configurations

#### Documentation:
- [ ] API documentation with examples
- [ ] Migration guide from current ServiceConfigLoader
- [ ] Best practices for configuration management
- [ ] Environment-specific setup guides

## Migration Strategy

### Phase 1: Backward Compatibility
```python
# Provide wrapper for existing ServiceConfigLoader usage
class LegacyConfigAdapter:
    @staticmethod
    def load_config(*args, **kwargs):
        """Compatibility wrapper for existing code."""
        return UnifiedConfigLoader.load_config(*args, **kwargs)
```

### Phase 2: Service-by-Service Migration
1. Update `drl-trading-common` with new loader
2. Migrate `drl-trading-inference` (simplest service)
3. Migrate `drl-trading-training`
4. Migrate remaining services
5. Remove legacy compatibility layer

## Acceptance Criteria
- [ ] New configuration loader handles all current ServiceConfigLoader use cases
- [ ] Schema validation works with all existing configuration classes
- [ ] Environment variable overrides work for nested configurations
- [ ] Secret substitution works with multiple pattern types
- [ ] Performance is equal or better than current implementation
- [ ] All existing services can migrate without breaking changes
- [ ] Comprehensive test coverage (>90%)
- [ ] Documentation complete with migration examples

## Dependencies
- **Depends On:** T001 - Architecture decisions must be finalized
- **Blocks:** T004 - Service Bootstrap Pattern Standardization

## Risks
- **Breaking Changes**: Migration could break existing services
  - **Mitigation**: Comprehensive backward compatibility layer
- **Performance Regression**: New implementation could be slower
  - **Mitigation**: Performance benchmarking and optimization
- **Complex Migration**: Services have different configuration patterns
  - **Mitigation**: Flexible adapter patterns and migration tools

## Definition of Done
- [ ] Implementation complete and tested
- [ ] Backward compatibility maintained for existing code
- [ ] Performance benchmarks show no regression
- [ ] All unit and integration tests pass
- [ ] Documentation complete with migration guide
- [ ] At least one service successfully migrated as proof-of-concept
- [ ] Code review completed and approved
