# Reusable Architecture Testing Framework

## For Each Service

### 1. Create Test File

`drl-trading-{service}/tests/unit/architecture/architecture_test.py`:

```python
from pathlib import Path
import pytest
from drl_trading_common.testing.architecture_test_base import (
    BaseArchitectureDocumentationTest,
    BaseArchitectureTest,
)

class Test{Service}Architecture(BaseArchitectureTest):
    service_name = "{service}"  # e.g., "training"
    service_package = "drl_trading_{service}"  # e.g., "drl_trading_training"

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

class Test{Service}ArchitectureDocumentation(BaseArchitectureDocumentationTest):
    service_name = "{service}"
    service_package = "drl_trading_{service}"
```

### 2. Add `.importlinter` Config

Already exists for preprocess - copy and adjust service name.

### 3. That's It!

Run tests:
```bash
pytest tests/unit/architecture/ -v
```

## Workspace-Level Testing

Test ALL services from workspace root:

```bash
# From workspace root
pytest tests/architecture_all_services_test.py -v

# Specific service only
pytest tests/architecture_all_services_test.py -v -k "preprocess"
```

Add new service: Edit `SERVICES` list in `tests/architecture_all_services_test.py`.

## What's Shared (in drl-trading-common)

1. **architecture_rules.py**: Rule definitions (parameterized by service name)
2. **architecture_test_base.py**: All test methods, just inherit and configure
3. **Zero duplication**: Only service name/package differs between services

## Files Per Service

```
drl-trading-{service}/
├── .importlinter                           # Config (copy from preprocess)
└── tests/unit/architecture/
    ├── __init__.py
    ├── architecture_test.py                # 30 lines - just service config
    └── README.md                           # Optional docs
```

## Example Files

- [architecture_test.py](architecture_test.py) - Working example for preprocess service
- [EXAMPLE_other_service.py](EXAMPLE_other_service.py) - Template for other services
- [tests/architecture_all_services_test.py](../../../../../../tests/architecture_all_services_test.py) - Workspace-level runner
- [drl-trading-common/src/drl_trading_common/testing/architecture_rules.py](drl-trading-common/src/drl_trading_common/testing/architecture_rules.py) - Shared rule definitions
- [drl-trading-common/src/drl_trading_common/testing/architecture_test_base.py](../../../../../../../drl-trading-common/src/drl_trading_common/testing/architecture_test_base.py) - Shared test base classes
