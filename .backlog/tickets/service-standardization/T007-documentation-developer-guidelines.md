# T007: Documentation & Developer Guidelines

**Priority:** High
**Estimated Effort:** 2 days
**Type:** Documentation
**Depends On:** T006

## Objective
Create comprehensive documentation and developer guidelines for the standardized service architecture, ensuring team adoption and providing clear guidance for future service development.

## Scope
Develop complete documentation covering:
- Service standardization architecture overview
- Developer guidelines for new service creation
- Configuration management best practices
- Secret management procedures
- Bootstrap pattern usage
- Logging standards and practices
- Troubleshooting guides
- Migration examples and templates

## Documentation Structure

### 1. Architecture Overview Documentation
**Duration:** 0.5 days

#### Service Standardization Architecture Guide
```markdown
# DRL Trading Service Standardization Architecture

## Overview
The DRL Trading platform uses a standardized service architecture that ensures consistency, maintainability, and operational excellence across all microservices.

## Core Principles
1. **Uniform Configuration**: All services use identical configuration patterns
2. **Secure by Default**: Built-in secret management and security practices
3. **Observable**: Consistent logging and health monitoring
4. **Resilient**: Graceful shutdown and error handling
5. **Developer Friendly**: Clear patterns reduce cognitive load

## Architecture Components

### Configuration Management
- **ServiceConfigLoader**: Unified configuration loading with environment detection
- **Secret Substitution**: `${VAR}` placeholder support with fallbacks
- **Environment-Specific Configs**: `.development.yaml`, `.production.yaml` variants
- **Schema Validation**: Pydantic-based configuration validation

### Service Bootstrap
- **ServiceBootstrap**: Abstract base class for service initialization
- **Dependency Injection**: Consistent DI container patterns
- **Graceful Shutdown**: SIGTERM/SIGINT handling
- **Health Checks**: Standardized health monitoring

### Logging Standards
- **ServiceLogger**: Unified logging configuration
- **Structured Logging**: JSON format for production environments
- **Context Tracking**: Correlation IDs and request tracing
- **Performance**: Log sampling for high-volume scenarios

### Security
- **Secret Management**: Environment-based secret injection
- **Least Privilege**: Services access only required secrets
- **Audit Trail**: Secret access logging (where applicable)

## Service Patterns

### Standard Service Structure
```
drl-trading-{service}/
├── config/
│   ├── {service}.development.yaml
│   ├── {service}.production.yaml
│   └── {service}.staging.yaml
├── src/
│   └── drl_trading_{service}/
│       ├── bootstrap/
│       │   └── {service}_bootstrap.py
│       ├── config/
│       │   └── {service}_config.py
│       ├── di/
│       │   └── {service}_module.py
│       └── ...
├── main.py
├── pyproject.toml
└── README.md
```

### Standard main.py Pattern
```python
from drl_trading_{service}.bootstrap.{service}_bootstrap import {Service}ServiceBootstrap

def main():
    """Main entry point for {service} service."""
    bootstrap = {Service}ServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()
```
```

#### Best Practices Guide
```markdown
# Service Development Best Practices

## Configuration Best Practices

### 1. Configuration Schema Design
- Use Pydantic models for all configuration classes
- Provide clear descriptions for all fields
- Set sensible defaults where appropriate
- Use Field validation for business rules

```python
class ServiceConfig(BaseApplicationConfig):
    \"\"\"Configuration for service.\"\"\"

    api_timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="API timeout in seconds"
    )

    batch_size: int = Field(
        default=100,
        ge=1,
        description="Processing batch size"
    )
```

### 2. Secret Management
- Never hardcode secrets in configuration files
- Use `${VAR}` placeholders for all sensitive data
- Provide defaults only for non-sensitive configurations
- Document required environment variables

```yaml
# config/service.yaml
database:
  host: ${DB_HOST:localhost}
  username: ${DB_USERNAME}
  password: ${DB_PASSWORD}  # No default for secrets
```

### 3. Environment-Specific Configuration
- Create separate config files for each environment
- Use environment-specific overrides sparingly
- Keep common configuration in base files
- Validate configuration in CI/CD pipelines

## Bootstrap Pattern Usage

### 1. Service Bootstrap Implementation
```python
class MyServiceBootstrap(ServiceBootstrap):
    def __init__(self):
        super().__init__(
            service_name="myservice",
            config_class=MyServiceConfig
        )

    def get_dependency_modules(self) -> List[Module]:
        return [
            MyServiceModule(self.config),
            CommonModule(self.config)
        ]

    def _start_service(self) -> None:
        self.service = self.injector.get(MyService)
        self.service.start()

    def _stop_service(self) -> None:
        if self.service:
            self.service.stop()

    def _run_main_loop(self) -> None:
        # Service-specific main loop
        pass
```

### 2. Dependency Injection Best Practices
- Use singleton pattern for stateful services
- Prefer constructor injection over field injection
- Create focused modules for different concerns
- Use interfaces for testability

## Logging Best Practices

### 1. Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Potential issues that don't prevent operation
- **ERROR**: Error conditions that might still allow operation
- **CRITICAL**: Serious errors that may abort the program

### 2. Structured Logging
```python
logger.info("Processing batch", extra={
    'batch_id': batch_id,
    'batch_size': len(items),
    'processing_time_ms': processing_time
})
```

### 3. Context Management
```python
with service_logger.correlation_context() as correlation_id:
    logger.info("Starting operation")
    # All logs in this context will include correlation_id
```
```

### 2. Developer Quick Start Guide
**Duration:** 0.5 days

#### New Service Creation Template
```markdown
# Creating a New DRL Trading Service

## Quick Start Checklist
- [ ] Create service directory structure
- [ ] Implement configuration schema
- [ ] Create service bootstrap class
- [ ] Set up dependency injection module
- [ ] Implement main.py entry point
- [ ] Create configuration files
- [ ] Add health checks
- [ ] Write integration tests
- [ ] Update documentation

## Step-by-Step Guide

### 1. Create Service Structure
```bash
# Generate service skeleton
./scripts/create-service.sh my-new-service

# This creates:
drl-trading-my-new-service/
├── config/
├── src/drl_trading_my_new_service/
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

### 2. Define Configuration Schema
```python
# src/drl_trading_my_new_service/config/my_new_service_config.py
from drl_trading_common.base.base_application_config import BaseApplicationConfig
from drl_trading_common.config.logging_config import LoggingConfig
from pydantic import Field

class MyNewServiceConfig(BaseApplicationConfig):
    \"\"\"Configuration for my new service.\"\"\"

    # Service-specific configuration
    worker_count: int = Field(default=4, description="Number of worker threads")
    timeout_seconds: int = Field(default=30, description="Operation timeout")

    # Standard configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
```

### 3. Implement Service Bootstrap
```python
# src/drl_trading_my_new_service/bootstrap/my_new_service_bootstrap.py
from drl_trading_common.bootstrap.service_bootstrap import ServiceBootstrap
from drl_trading_my_new_service.config.my_new_service_config import MyNewServiceConfig

class MyNewServiceBootstrap(ServiceBootstrap):
    def __init__(self):
        super().__init__(
            service_name="my-new-service",
            config_class=MyNewServiceConfig
        )
```

### 4. Create Configuration Files
```yaml
# config/my-new-service.development.yaml
worker_count: 2
timeout_seconds: 10

logging:
  level: DEBUG
  json_format: false

# Development secrets (use .env file)
api_key: ${API_KEY:dev_api_key}
```

### 5. Add to Docker Compose
```yaml
# docker-compose.yml
services:
  my-new-service:
    build:
      context: ./drl-trading-my-new-service
    environment:
      - DEPLOYMENT_MODE=development
    volumes:
      - ./drl-trading-my-new-service:/app
```
```

#### Service Template Generator
```python
# scripts/create_service_template.py
import os
import argparse
from pathlib import Path

def create_service_template(service_name: str) -> None:
    """Generate a new service template with standardized structure."""

    service_dir = f"drl-trading-{service_name}"

    # Create directory structure
    directories = [
        f"{service_dir}/config",
        f"{service_dir}/src/drl_trading_{service_name}/bootstrap",
        f"{service_dir}/src/drl_trading_{service_name}/config",
        f"{service_dir}/src/drl_trading_{service_name}/di",
        f"{service_dir}/tests/unit",
        f"{service_dir}/tests/integration",
        f"{service_dir}/logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Generate template files
    _generate_main_py(service_dir, service_name)
    _generate_bootstrap(service_dir, service_name)
    _generate_config(service_dir, service_name)
    _generate_di_module(service_dir, service_name)
    _generate_config_files(service_dir, service_name)
    _generate_pyproject_toml(service_dir, service_name)

    print(f"✅ Service template created: {service_dir}")
    print("Next steps:")
    print("1. Implement service-specific logic")
    print("2. Update configuration schema")
    print("3. Add health checks")
    print("4. Write tests")
    print("5. Update documentation")

def _generate_main_py(service_dir: str, service_name: str) -> None:
    """Generate main.py file."""
    class_name = ''.join(word.capitalize() for word in service_name.split('-'))
    content = f"""from drl_trading_{service_name}.bootstrap.{service_name}_bootstrap import {class_name}ServiceBootstrap

def main():
    \"\"\"Main entry point for {service_name} service.\"\"\"
    bootstrap = {class_name}ServiceBootstrap()
    bootstrap.start()

if __name__ == "__main__":
    main()
"""
    with open(f"{service_dir}/main.py", "w") as f:
        f.write(content)
```

### 3. Configuration Management Guide
**Duration:** 0.5 days

#### Configuration Best Practices Guide
```markdown
# Configuration Management Guide

## Configuration Loading Patterns

### Basic Configuration Loading
```python
from drl_trading_common.config.service_config_loader import ServiceConfigLoader
from my_service.config.my_service_config import MyServiceConfig

# Load configuration
config = ServiceConfigLoader.load_config(
    config_class=MyServiceConfig,
    service="my-service"
)
```

### Environment-Specific Configuration
```python
# Automatic environment detection
config = ServiceConfigLoader.load_config(
    config_class=MyServiceConfig,
    service="my-service"
    # Will load my-service.{DEPLOYMENT_MODE}.yaml
)

# Explicit configuration file
config = ServiceConfigLoader.load_config(
    config_class=MyServiceConfig,
    config_path="config/my-service.production.yaml"
)
```

### Secret Management Patterns

#### Development Environment (.env)
```bash
# .env
DB_PASSWORD=dev_password
API_KEY=dev_api_key
REDIS_URL=redis://localhost:6379
```

#### Production Environment (Kubernetes)
```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: my-service-secrets
data:
  db-password: <base64-encoded>
  api-key: <base64-encoded>

# k8s/deployment.yaml
env:
  - name: DB_PASSWORD
    valueFrom:
      secretKeyRef:
        name: my-service-secrets
        key: db-password
```

## Configuration Validation

### Schema Validation Examples
```python
from pydantic import Field, validator

class DatabaseConfig(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(5432, ge=1, le=65535)

    @validator('host')
    def validate_host(cls, v):
        if not v or v.isspace():
            raise ValueError('Host cannot be empty')
        return v.strip()

class ApiConfig(BaseModel):
    timeout: int = Field(30, ge=1, le=300)
    retry_attempts: int = Field(3, ge=0, le=10)
```

### Configuration Testing
```python
class TestMyServiceConfig:
    def test_valid_configuration(self):
        \"\"\"Test that valid configuration loads correctly.\"\"\"
        # Given
        config_data = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"timeout": 30}
        }

        # When
        config = MyServiceConfig(**config_data)

        # Then
        assert config.database.host == "localhost"
        assert config.api.timeout == 30

    def test_invalid_configuration_raises_error(self):
        \"\"\"Test that invalid configuration raises validation error.\"\"\"
        # Given
        config_data = {"api": {"timeout": -1}}  # Invalid timeout

        # When/Then
        with pytest.raises(ValidationError):
            MyServiceConfig(**config_data)
```
```

### 4. Troubleshooting Guide
**Duration:** 0.5 days

#### Common Issues and Solutions
```markdown
# Service Standardization Troubleshooting Guide

## Configuration Issues

### Issue: Service fails to start with "Configuration file not found"
**Symptoms:**
```
FileNotFoundError: No configuration file found for service my-service
```

**Solutions:**
1. Check configuration file exists in expected location:
   ```bash
   ls config/my-service.development.yaml
   ```

2. Verify DEPLOYMENT_MODE environment variable:
   ```bash
   echo $DEPLOYMENT_MODE
   ```

3. Use explicit config path:
   ```python
   config = ServiceConfigLoader.load_config(
       config_class=MyServiceConfig,
       config_path="path/to/config.yaml"
   )
   ```

### Issue: Secret substitution not working
**Symptoms:**
```
Configuration contains literal "${DB_PASSWORD}" instead of actual password
```

**Solutions:**
1. Check environment variable is set:
   ```bash
   echo $DB_PASSWORD
   ```

2. Verify .env file is loaded:
   ```bash
   cat .env | grep DB_PASSWORD
   ```

3. Use explicit secret provider configuration:
   ```python
   providers = [
       EnvironmentSecretProvider(),
       FileSecretProvider(".env")
   ]
   ```

## Bootstrap Issues

### Issue: Service hangs during startup
**Symptoms:**
- Service starts but never completes initialization
- No error messages in logs

**Solutions:**
1. Enable debug logging:
   ```yaml
   logging:
     level: DEBUG
   ```

2. Check dependency injection configuration:
   ```python
   # Verify all required dependencies are bound
   def get_dependency_modules(self) -> List[Module]:
       return [RequiredModule(self.config)]
   ```

3. Add timeout to service startup:
   ```python
   def start(self, timeout: int = 30):
       # Add startup timeout logic
   ```

### Issue: Graceful shutdown not working
**Symptoms:**
- Service doesn't respond to SIGTERM
- Docker containers must be force-killed

**Solutions:**
1. Verify signal handlers are registered:
   ```python
   def _setup_signal_handlers(self) -> None:
       signal.signal(signal.SIGTERM, self._signal_handler)
       signal.signal(signal.SIGINT, self._signal_handler)
   ```

2. Check main loop implementation:
   ```python
   def _run_main_loop(self) -> None:
       while self.is_running:  # Must check this flag
           time.sleep(1)
   ```

## Logging Issues

### Issue: Logs not appearing in expected format
**Symptoms:**
- Development environment shows JSON logs
- Production environment shows human-readable logs

**Solutions:**
1. Check environment detection:
   ```python
   print(os.environ.get("DEPLOYMENT_MODE"))
   ```

2. Force JSON format:
   ```yaml
   logging:
     json_format: true
   ```

### Issue: Correlation IDs not appearing in logs
**Symptoms:**
- Logs don't contain correlation_id field
- Cross-service tracing doesn't work

**Solutions:**
1. Use correlation context:
   ```python
   with service_logger.correlation_context() as correlation_id:
       logger.info("Operation started")
   ```

2. Verify structured logging is enabled:
   ```python
   # Check formatter type
   handler = logger.handlers[0]
   print(type(handler.formatter))
   ```

## Dependency Injection Issues

### Issue: "No binding found for interface"
**Symptoms:**
```
InjectorError: No binding found for interface MyInterface
```

**Solutions:**
1. Check module binding:
   ```python
   class MyModule(Module):
       def configure(self, binder):
           binder.bind(MyInterface, to=MyImplementation, scope=singleton)
   ```

2. Verify module is included:
   ```python
   def get_dependency_modules(self) -> List[Module]:
       return [MyModule(self.config)]  # Include the module
   ```

## Performance Issues

### Issue: Service startup is slow
**Symptoms:**
- Service takes > 10 seconds to start
- Health checks timing out

**Solutions:**
1. Profile startup sequence:
   ```python
   import time

   start = time.time()
   self._load_configuration()
   print(f"Config loading: {time.time() - start}s")
   ```

2. Optimize dependency injection:
   ```python
   # Use lazy loading for expensive resources
   @provider
   @singleton
   def provide_expensive_service(self) -> ExpensiveService:
       return ExpensiveService()  # Only created when needed
   ```

## Migration Issues

### Issue: Service functionality broken after migration
**Symptoms:**
- Previously working features no longer work
- Different behavior compared to pre-migration

**Solutions:**
1. Compare configurations:
   ```bash
   diff config/old-config.json config/new-service.yaml
   ```

2. Check service dependencies:
   ```python
   # Verify all required services are injected
   def get_dependency_modules(self) -> List[Module]:
       return [AllRequiredModules()]
   ```

3. Rollback if necessary:
   ```bash
   ./scripts/rollback_service_migration.py my-service
   ```
```

## Testing Documentation

### Integration Test Examples
```python
# Example integration test for standardized service
class TestStandardizedServiceIntegration:
    def test_service_startup_and_health(self):
        \"\"\"Test that service starts and reports healthy.\"\"\"
        # Given
        bootstrap = MyServiceBootstrap()

        # When
        bootstrap.start()

        # Then
        assert bootstrap.is_running
        health_status = bootstrap.injector.get(HealthCheckService).check_health()
        assert health_status['status'] == 'healthy'

        # Cleanup
        bootstrap.stop()

    def test_configuration_loading(self):
        \"\"\"Test that configuration loads correctly.\"\"\"
        # Given
        config_path = "tests/resources/test-config.yaml"

        # When
        config = ServiceConfigLoader.load_config(
            config_class=MyServiceConfig,
            config_path=config_path
        )

        # Then
        assert config is not None
        assert config.worker_count > 0

    def test_secret_substitution(self):
        \"\"\"Test that secret substitution works.\"\"\"
        # Given
        os.environ["TEST_SECRET"] = "secret_value"
        config_dict = {"api_key": "${TEST_SECRET}"}

        # When
        substitution = SecretSubstitution([EnvironmentSecretProvider()])
        result = substitution.substitute(config_dict)

        # Then
        assert result["api_key"] == "secret_value"
```

## Acceptance Criteria
- [ ] Complete architecture documentation published
- [ ] Developer quick start guide with working examples
- [ ] Configuration management guide with best practices
- [ ] Troubleshooting guide covering common issues
- [ ] Service template generator script functional
- [ ] Integration test examples provided
- [ ] Migration examples documented
- [ ] API reference documentation complete
- [ ] Performance guidelines established
- [ ] Security best practices documented

## Dependencies
- **Depends On:** T006 (Service Migration - needs real examples)
- **Blocks:** Future service development (provides templates and guidelines)

## Risks
- **Documentation Drift**: Documentation becomes outdated as system evolves
  - **Mitigation**: Include documentation updates in definition of done for future changes
- **Adoption Resistance**: Developers might not follow guidelines
  - **Mitigation**: Clear examples, templates, and code review enforcement
- **Complexity Overwhelm**: Too much documentation might be ignored
  - **Mitigation**: Progressive disclosure with quick start guides

## Definition of Done
- [ ] Architecture overview documentation complete and published
- [ ] Developer quick start guide tested with new developer
- [ ] Configuration management guide covers all common scenarios
- [ ] Troubleshooting guide addresses all known issues from migration
- [ ] Service template generator creates working service skeleton
- [ ] Integration test examples work and are documented
- [ ] Performance guidelines established with benchmarks
- [ ] Security best practices documented and validated
- [ ] All documentation reviewed and approved by team
- [ ] Documentation integrated into development workflow
