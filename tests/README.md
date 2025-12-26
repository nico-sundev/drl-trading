# Workspace-Level Architecture Tests

This directory contains tests that can validate all services from the workspace root.

## Running All Services

```bash
# Test all services at once
pytest tests/architecture_all_services_test.py -v

# Test specific service
pytest tests/architecture_all_services_test.py -v -k "preprocess"

# Show detailed violations
pytest tests/architecture_all_services_test.py -v --tb=short
```

## Adding New Services

Edit `architecture_all_services_test.py` and add to the `SERVICES` list:

```python
SERVICES = [
    # ... existing services ...
    {"name": "my_service", "package": "drl_trading_my_service", "path": "drl-trading-my-service"},
]
```

That's it! No other code needed.
