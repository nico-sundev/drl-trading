# ADR-0002: Configuration Architecture Standardization

**Date:** 2025-08-02
**Status:** Accepted
**Tags:** configuration, microservices, infrastructure

## Context and Problem Statement

The DRL trading system currently has inconsistent configuration management across microservices. Different services use different formats (JSON vs YAML) and there's no standardized approach for environment-specific configurations, secret management, or schema validation.

**Current State Analysis:**
- **YAML Usage**: drl-trading-inference, drl-trading-ingest (newer services)
- **JSON Usage**: drl-trading-strategy-example (older service)
- **Custom Solution**: EnhancedServiceConfigLoader + ConfigAdapter (existing, working)
- **Schema Validation**: Pydantic BaseSchema across all services
- **Mixed Patterns**: No standardized approach to environment overrides

## Decision Drivers

- **User Preference**: YAML for readability and comment support
- **Ecosystem Alignment**: Docker, Kubernetes, and most DevOps tools use YAML
- **Schema Validation**: Must maintain current Pydantic validation
- **Environment Support**: Need robust environment-specific configuration
- **Secret Management**: Secure handling of sensitive configuration
- **Backwards Compatibility**: Minimize disruption to existing services

## Considered Options

### Option 1: Enhance Existing EnhancedServiceConfigLoader + YAML Standardization
- **Description**: Upgrade current custom solution with YAML-first approach
- **Pros**:
  - Minimal migration effort for existing services
  - Already proven to work with Pydantic
  - Full control over features and behavior
  - Deep integration with current BaseApplicationConfig pattern
- **Cons**:
  - Custom maintenance burden
  - Missing advanced features like secret substitution
  - No ecosystem tool integration

### Option 2: Pydantic Settings v2 with YAML Support
- **Description**: Use pydantic-settings with custom YAML loading
```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file='config/inference.yaml',
        env_prefix='INFERENCE_',
        case_sensitive=False,
        env_nested_delimiter='__'
    )
```
- **Pros**:
  - Official Pydantic ecosystem tool
  - Excellent environment variable support
  - Type validation and conversion
  - Nested field support with delimiters
- **Cons**:
  - Limited YAML support (requires custom loader)
  - No built-in secret management
  - No template/substitution features

### Option 3: Dynaconf with YAML Focus
- **Description**: Use Dynaconf for configuration management
```python
from dynaconf import Dynaconf

settings = Dynaconf(
    environments=True,
    settings_files=['config.yaml', 'config.local.yaml'],
    env_switcher='ENV_FOR_DYNACONF',
    merge_enabled=True
)
```
- **Pros**:
  - Multi-environment support out of the box
  - YAML native support with merging
  - Vault/Redis integration for secrets
  - Template variable substitution
- **Cons**:
  - Would require significant refactoring of Pydantic schemas
  - Different configuration patterns than current BaseApplicationConfig
  - Learning curve for team

### Option 4: Hydra (Facebook) Configuration Framework
- **Description**: Use Hydra for structured configuration
```python
from hydra import compose, initialize
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="inference")
def main(cfg: DictConfig) -> None:
    config = InferenceConfig(**cfg)
```
- **Pros**:
  - Powerful composition and override features
  - Excellent YAML support with OmegaConf
  - Command-line override integration
  - Advanced templating and interpolation
- **Cons**:
  - Heavyweight for microservice configuration
  - Designed more for ML experiments than production services
  - Would require major architectural changes

## Decision Outcome

**Chosen Option: Enhanced EnhancedServiceConfigLoader + YAML Standardization**

### Rationale:
1. **Minimal Migration Risk**: Builds on proven existing solution
2. **YAML Standardization**: Aligns with user preference and ecosystem
3. **Pydantic Integration**: Maintains current schema validation approach
4. **Incremental Enhancement**: Can add features (secrets, templates) as needed
5. **Service Compatibility**: Works with existing BaseApplicationConfig pattern

### Implementation Plan:

#### Phase 1: YAML Standardization (1-2 days)
- Standardize all services to use YAML configuration format
- Migrate existing JSON configs to YAML equivalents
- Update EnhancedServiceConfigLoader to prefer YAML by default

#### Phase 2: Enhanced Environment Support (2-3 days)
- Improve environment variable override patterns
- Add support for nested configuration overrides
- Implement environment-specific configuration templates

#### Phase 3: Secret Management Integration (3-4 days)
- Add secret substitution support for sensitive values
- Integrate with environment variable-based secret injection
- Support for external secret management systems (future)

#### Phase 4: Validation & Documentation (1-2 days)
- Enhanced validation error messages
- Configuration schema documentation generation
- Developer guidelines and examples

## Implementation Details

### Enhanced EnhancedServiceConfigLoader Features:
```python
class EnhancedServiceConfigLoader:
    @staticmethod
    def load_config(
        config_class: Type[T],
        service: Optional[str] = None,
        config_path: Optional[str] = None,
        env_override: bool = True,
        secret_substitution: bool = True
    ) -> T:
        # Enhanced implementation with YAML preference
        # Environment variable override support
        # Secret substitution for sensitive values
```

### YAML Configuration Structure:
```yaml
# service.yaml - Standard structure
app_name: "drl-trading-inference"
version: "1.0.0"
deployment_mode: "development"

infrastructure:
  service_name: "drl-trading-inference"
  logging:
    level: "INFO"
    file_path: "logs/service.log"

model_config:
  model_path: "${MODEL_PATH:/app/models/latest}"
  batch_size: 1

# Environment-specific overrides in service.production.yaml
infrastructure:
  logging:
    level: "WARNING"
    file_path: "/var/log/drl-trading-inference.log"
```

## Positive Consequences
- **Standardized Format**: All services use YAML for consistency
- **Improved Readability**: YAML supports comments and human-friendly structure
- **Environment Flexibility**: Robust support for different deployment environments
- **Backwards Compatibility**: Existing Pydantic schemas continue working
- **Incremental Enhancement**: Can add advanced features without major rewrites

## Negative Consequences
- **Migration Effort**: Need to convert existing JSON configs to YAML
- **Custom Maintenance**: Still maintaining custom configuration solution
- **Feature Parity**: Advanced features require custom implementation

## Validation Criteria
- [ ] All services successfully load configuration from YAML files
- [ ] Environment-specific overrides work correctly across development/production
- [ ] Pydantic schema validation continues working without changes
- [ ] Configuration loading performance remains acceptable
- [ ] Developer documentation is clear and comprehensive

## References
- [EnhancedServiceConfigLoader Implementation](../../drl-trading-common/src/drl_trading_common/config/enhanced_service_config_loader.py)
- [ConfigAdapter Implementation](../../drl-trading-common/src/drl_trading_common/config/config_adapter.py)
- [T001 Configuration Research Ticket](../../.backlog/tickets/service-standardization/T001-configuration-architecture-research.md)
- [T002 ServiceConfig Enhancement Ticket](../../.backlog/tickets/service-standardization/T002-serviceconfig-enhancement.md)
