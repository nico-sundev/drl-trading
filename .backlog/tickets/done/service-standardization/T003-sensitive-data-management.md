# T003: Sensitive Data Management Implementation

**Priority:** High
**Estimated Effort:** 3 days
**Type:** Implementation
**Depends On:** T001, T002

## Objective
Implement secure and standardized sensitive data management across all DRL trading microservices.

## Scope
Create a unified approach for handling secrets, API keys, database passwords, and other sensitive configuration data that works consistently across development, staging, and production environments.

## Requirements

### Security Requirements
- **No Secrets in Code**: No hardcoded secrets in any configuration files or source code
- **Environment Isolation**: Development secrets don't impact production
- **Audit Trail**: Track secret access and usage (where possible)
- **Rotation Support**: Easy secret rotation without service downtime
- **Least Privilege**: Services only access secrets they need

### Operational Requirements
- **Developer Experience**: Easy local development setup
- **CI/CD Integration**: Secure secret injection in pipelines
- **Container Compatibility**: Works with Docker and Kubernetes
- **Multi-Environment**: Supports dev/staging/production patterns
- **Fallback Handling**: Graceful degradation when secrets unavailable

## Implementation Design

### 1. Secret Substitution Engine
**Duration:** 1.5 days

#### Core Implementation:
```python
# drl_trading_common/config/secret_manager.py
from abc import ABC, abstractmethod
import os
import re
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SecretProvider(ABC):
    """Abstract base for secret providers."""

    @abstractmethod
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value."""
        pass

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secret keys."""
        pass

class EnvironmentSecretProvider(SecretProvider):
    """Provides secrets from environment variables."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        full_key = f"{self.prefix}{key}" if self.prefix else key
        return os.environ.get(full_key, default)

class FileSecretProvider(SecretProvider):
    """Provides secrets from .env files."""

    def __init__(self, env_file: str = ".env"):
        self.secrets = {}
        self._load_env_file(env_file)

    def _load_env_file(self, env_file: str):
        """Load secrets from .env file."""
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        self.secrets[key] = value

class SecretSubstitution:
    """Handles secret substitution in configuration."""

    def __init__(self, providers: List[SecretProvider]):
        self.providers = providers
        # Matches ${VAR}, ${VAR:default}, ${VAR:-default}
        self.pattern = re.compile(r'\$\{([^}:]+)(:?-?([^}]*))?\}')

    def substitute(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively substitute secrets in configuration dictionary."""
        return self._recursive_substitute(config_dict)

    def _recursive_substitute(self, obj: Any) -> Any:
        """Recursively substitute secrets in nested structures."""
        if isinstance(obj, dict):
            return {k: self._recursive_substitute(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._recursive_substitute(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string(obj)
        else:
            return obj

    def _substitute_string(self, value: str) -> str:
        """Substitute secrets in a string value."""
        def replace_match(match):
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            # Try each provider in order
            for provider in self.providers:
                secret_value = provider.get_secret(var_name)
                if secret_value is not None:
                    return secret_value

            # No provider had the secret
            if has_default:
                return default_value or ""
            else:
                logger.warning(f"Secret '{var_name}' not found and no default provided")
                return match.group(0)  # Return original placeholder

        return self.pattern.sub(replace_match, value)
```

#### Usage Example:
```yaml
# config/inference.yaml
database:
  host: ${DB_HOST:localhost}
  port: ${DB_PORT:5432}
  username: ${DB_USERNAME}
  password: ${DB_PASSWORD}

api:
  key: ${API_KEY}
  secret: ${API_SECRET}

redis:
  url: ${REDIS_URL:redis://localhost:6379}
```

### 2. Environment-Specific Secret Management
**Duration:** 1 day

#### Development Environment (.env files):
```bash
# .env.development
DB_HOST=localhost
DB_USERNAME=dev_user
DB_PASSWORD=dev_password
API_KEY=dev_api_key
API_SECRET=dev_api_secret
REDIS_URL=redis://localhost:6379
```

#### Production Environment (External Providers):
```python
# drl_trading_common/config/external_secret_providers.py

class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault integration."""

    def __init__(self, vault_url: str, token: str, mount_path: str = "secret"):
        self.vault_url = vault_url
        self.token = token
        self.mount_path = mount_path
        # Initialize Vault client

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Vault API integration
        pass

class AWSSecretsProvider(SecretProvider):
    """AWS Secrets Manager integration."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        # Initialize AWS client

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # AWS Secrets Manager API integration
        pass

class KubernetesSecretProvider(SecretProvider):
    """Kubernetes secrets integration."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.secrets_path = "/var/run/secrets"

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        # Read from mounted Kubernetes secrets
        secret_file = os.path.join(self.secrets_path, key)
        if os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                return f.read().strip()
        return default
```

### 3. Configuration Integration
**Duration:** 0.5 days

#### Enhanced Config Loader Integration:
```python
# Integration with T002 configuration loader
class SecretAwareConfigLoader(BaseConfigLoader):
    def __init__(self):
        self.secret_substitution = self._create_secret_substitution()

    def _create_secret_substitution(self) -> SecretSubstitution:
        """Create secret substitution based on environment."""
        providers = []

        # Always include environment variables
        providers.append(EnvironmentSecretProvider())

        # Add environment-specific providers
        env = os.environ.get("DEPLOYMENT_MODE", "development")

        if env == "development":
            providers.append(FileSecretProvider(".env"))
            providers.append(FileSecretProvider(f".env.{env}"))
        elif env == "production":
            # Add production secret providers
            if os.environ.get("VAULT_ADDR"):
                providers.append(VaultSecretProvider(
                    vault_url=os.environ["VAULT_ADDR"],
                    token=os.environ["VAULT_TOKEN"]
                ))
            if os.path.exists("/var/run/secrets"):
                providers.append(KubernetesSecretProvider())

        return SecretSubstitution(providers)

    def load_config(self, config_class: Type[T], **kwargs) -> T:
        """Load configuration with secret substitution."""
        # Load raw configuration
        config_dict = super()._load_raw_config(**kwargs)

        # Apply secret substitution
        substituted_config = self.secret_substitution.substitute(config_dict)

        # Validate and return typed configuration
        return config_class(**substituted_config)
```

## Security Best Practices

### 1. Secret Management Guidelines
```python
# drl_trading_common/config/security_guidelines.py
class SecretValidation:
    """Security validation for secrets."""

    @staticmethod
    def validate_secret_strength(secret: str, min_length: int = 12) -> bool:
        """Validate secret meets minimum security requirements."""
        if len(secret) < min_length:
            return False
        # Add additional validation logic
        return True

    @staticmethod
    def mask_secret_for_logging(secret: str) -> str:
        """Mask secret for safe logging."""
        if len(secret) <= 4:
            return "*" * len(secret)
        return secret[:2] + "*" * (len(secret) - 4) + secret[-2:]
```

### 2. Development Environment Setup
**Create setup script:**
```bash
#!/bin/bash
# scripts/setup-dev-secrets.sh

echo "Setting up development environment secrets..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
    echo "Please update .env with your development secrets"
fi

# Validate required secrets exist
python scripts/validate-secrets.py
```

```python
# scripts/validate-secrets.py
import os
from pathlib import Path

REQUIRED_SECRETS = [
    "DB_PASSWORD",
    "API_KEY",
    "API_SECRET"
]

def validate_development_secrets():
    """Validate all required secrets are available."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False

    # Load .env and check required secrets
    missing_secrets = []
    with open(env_file) as f:
        env_vars = {}
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                env_vars[key] = value

    for secret in REQUIRED_SECRETS:
        if secret not in env_vars or not env_vars[secret]:
            missing_secrets.append(secret)

    if missing_secrets:
        print(f"❌ Missing required secrets: {', '.join(missing_secrets)}")
        return False

    print("✅ All required secrets are configured")
    return True

if __name__ == "__main__":
    validate_development_secrets()
```

## Environment-Specific Implementation

### Development Environment
```yaml
# .env.example (checked into repository)
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=dev_user
DB_PASSWORD=change_me_in_dot_env

# API Configuration
API_KEY=your_development_api_key
API_SECRET=your_development_api_secret

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Feature Store Configuration
S3_ACCESS_KEY=minio_access_key
S3_SECRET_KEY=minio_secret_key
```

### Docker Compose Development
```yaml
# docker-compose.development.yml
version: '3.8'
services:
  inference:
    environment:
      - DB_PASSWORD=docker_dev_password
      - API_KEY=docker_dev_key
    env_file:
      - .env
```

### Kubernetes Production
```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: drl-trading-secrets
type: Opaque
data:
  db-password: <base64-encoded-password>
  api-key: <base64-encoded-key>
  api-secret: <base64-encoded-secret>
```

## Testing Strategy

### Unit Tests
```python
class TestSecretSubstitution:
    def test_simple_substitution(self):
        """Test basic secret substitution."""
        # Given
        providers = [EnvironmentSecretProvider()]
        substitution = SecretSubstitution(providers)
        config = {"database": {"password": "${DB_PASSWORD}"}}

        # When
        os.environ["DB_PASSWORD"] = "test_password"
        result = substitution.substitute(config)

        # Then
        assert result["database"]["password"] == "test_password"

    def test_default_value_substitution(self):
        """Test substitution with default values."""
        # Given
        providers = [EnvironmentSecretProvider()]
        substitution = SecretSubstitution(providers)
        config = {"redis": {"url": "${REDIS_URL:redis://localhost:6379}"}}

        # When
        result = substitution.substitute(config)

        # Then
        assert result["redis"]["url"] == "redis://localhost:6379"
```

## Acceptance Criteria
- [ ] Secret substitution works with `${VAR}` and `${VAR:default}` syntax
- [ ] Multiple secret providers can be configured in priority order
- [ ] Development environment uses .env files seamlessly
- [ ] Production environment integrates with external secret management
- [ ] No secrets are logged or exposed in error messages
- [ ] Secret validation prevents weak/empty secrets in production
- [ ] All existing service configurations can adopt secret management
- [ ] Comprehensive test coverage for security scenarios
- [ ] Documentation includes security best practices and setup guides

## Dependencies
- **Depends On:** T001 (Architecture decisions), T002 (Config loader implementation)
- **Blocks:** T004 (Service bootstrapping needs secret management)

## Risks
- **Secret Exposure**: Implementation bugs could expose secrets
  - **Mitigation**: Comprehensive security testing and code review
- **Development Complexity**: Too complex setup discourages adoption
  - **Mitigation**: Simple .env file setup with clear documentation
- **Production Integration**: External secret management integration issues
  - **Mitigation**: Fallback mechanisms and thorough testing

## Definition of Done
- [x] Secret substitution engine implemented and tested
- [x] Multiple secret providers implemented (env vars, .env files, external)
- [x] Integration with configuration loader complete
- [x] Security validation and masking implemented
- [x] Development environment setup scripts created
- [x] Production deployment patterns documented
- [x] Comprehensive security testing completed
- [x] Code review focused on security aspects completed

## Status: COMPLETED ✅

**Completion Date:** August 5, 2025
**Implementation Notes:**
- Secret substitution already implemented in `EnhancedServiceConfigLoader` with `${VAR:default}` syntax
- STAGE environment variable standardized across all services (local/cicd/prod)
- .env.example templates created for all six services
- Base configuration architecture simplified (eliminated deployment_mode redundancy)
- All requirements met and validated
