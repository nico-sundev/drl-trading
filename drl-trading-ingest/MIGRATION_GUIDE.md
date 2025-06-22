# Database Migration Integration

This document describes the Alembic migration system integration for the DRL Trading Ingest service.

## Overview

The service now uses Alembic for database schema management, following hexagonal architecture principles with clean separation of concerns:

- **Migration Service Interface** (Port): Defines the contract for migration operations
- **Alembic Migration Service** (Adapter): Implements migrations using Alembic
- **TimescaleRepo** (Updated): Focuses only on data operations, no longer handles schema

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  main.py                                                    │
│  ├─ Startup Migration Check                                 │
│  └─ Service Initialization                                  │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer (Ports)                     │
├─────────────────────────────────────────────────────────────┤
│  MigrationServiceInterface                                  │
│  ├─ migrate_to_latest()                                     │
│  ├─ create_migration()                                      │
│  ├─ get_current_revision()                                  │
│  └─ ensure_migrations_applied()                             │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure (Adapters)                 │
├─────────────────────────────────────────────────────────────┤
│  AlembicMigrationService                                    │
│  ├─ Alembic Command Integration                             │
│  ├─ Configuration Management                                │
│  └─ Error Handling                                          │
├─────────────────────────────────────────────────────────────┤
│  TimescaleRepo (Updated)                                    │
│  ├─ Data Operations Only                                    │
│  ├─ No Schema Management                                    │
│  └─ Proper Error Handling                                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Automatic Migration on Startup**
The service automatically applies pending migrations when starting up:

```python
# main.py
migration_service = injector.get(MigrationServiceInterface)
migration_service.ensure_migrations_applied()
```

### 2. **Configuration-Driven Connection**
Database connection is built from the injected configuration:

```python
connection_string = (
    f"postgresql://{db_config.username}:{db_config.password}"
    f"@{db_config.host}:{db_config.port}/{db_config.database}"
)
```

### 3. **Dependency Injection Integration**
Migration service is properly integrated with the DI container:

```python
# IngestModule
@provider
@singleton
def provide_migration_service(self, config: DataIngestionConfig) -> MigrationServiceInterface:
    return AlembicMigrationService(config)
```

### 4. **CLI Management Tools**
Command-line interface for manual migration management:

```bash
# Apply all pending migrations
python -m drl_trading_ingest.adapter.cli.migration_cli migrate

# Create new migration
python -m drl_trading_ingest.adapter.cli.migration_cli create-migration "Add new table"

# Check migration status
python -m drl_trading_ingest.adapter.cli.migration_cli status
```

## Usage

### Development Setup

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Set Configuration**:
   ```bash
   export SERVICE_CONFIG_PATH=/path/to/config.json
   ```

3. **Initialize Migration Repository** (first time only):
   ```bash
   python -m drl_trading_ingest.adapter.cli.migration_cli init
   ```

4. **Apply Migrations**:
   ```bash
   python -m drl_trading_ingest.adapter.cli.migration_cli migrate
   ```

### Creating New Migrations

1. **Modify Database Schema** (edit models or write SQL)

2. **Generate Migration**:
   ```bash
   python -m drl_trading_ingest.adapter.cli.migration_cli create-migration "Description of changes"
   ```

3. **Review Generated Migration** in `migrations/versions/`

4. **Apply Migration**:
   ```bash
   python -m drl_trading_ingest.adapter.cli.migration_cli migrate
   ```

### Production Deployment

1. **Automatic Application**: Migrations are applied automatically on service startup

2. **Manual Control**: Use CLI commands for manual migration management

3. **Rollback Support**: Use `migrate-to <revision>` for targeted rollbacks

## Migration Files

### Initial Migration (`001_initial_tables.py`)

Creates the foundational schema for market data storage:

- **market_data_eurusd**: Example table with TimescaleDB hypertable
- **Proper Indexing**: Optimized for time-series queries
- **Conflict Resolution**: Handles duplicate timestamps

### Future Migrations

New migrations should follow the pattern:
- **Descriptive Names**: Clear indication of changes
- **Proper Upgrade/Downgrade**: Support for rollbacks
- **TimescaleDB Optimization**: Leverage hypertables and indexing

## Error Handling

### Migration Errors
- **Wrapped in MigrationError**: Consistent error handling
- **Detailed Logging**: Complete error context
- **Graceful Degradation**: Service behavior on migration failure

### Database Connection
- **Connection Validation**: Verify database connectivity
- **Configuration Errors**: Clear error messages for misconfigurations
- **Retry Logic**: Implemented at the application level

## Testing

### Unit Tests
- **Interface Compliance**: Verify implementation matches interface
- **Error Scenarios**: Test failure handling
- **Configuration Validation**: Test config parsing

### Integration Tests
- **Database Integration**: Test with real database connections
- **Migration Application**: Verify migrations work correctly
- **CLI Commands**: Test command-line interface

### Running Tests
```bash
# Unit tests
pytest tests/unit/migration/

# Integration tests
pytest tests/integration/migration/

# All migration tests
pytest tests/ -k migration
```

## Best Practices

### Migration Development
1. **Review Generated Migrations**: Always check auto-generated files
2. **Test Rollbacks**: Ensure downgrade functions work
3. **Data Preservation**: Consider existing data when changing schema
4. **Performance Impact**: Evaluate migration performance on large datasets

### Production Considerations
1. **Backup First**: Always backup production data before migrations
2. **Maintenance Windows**: Schedule migrations during low-traffic periods
3. **Monitoring**: Monitor migration application in production
4. **Rollback Plan**: Have a clear rollback strategy

## Configuration

### Environment Variables
- `SERVICE_CONFIG_PATH`: Path to service configuration file

### Database Configuration
Migration service uses the same database configuration as the main service:

```json
{
  "infrastructure": {
    "database": {
      "host": "localhost",
      "port": 5432,
      "username": "postgres",
      "password": "password",
      "database": "marketdata"
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **"alembic.ini not found"**
   - Ensure alembic.ini exists in service root
   - Check file permissions

2. **"Migration repository not initialized"**
   - Run `python -m drl_trading_ingest.adapter.cli.migration_cli init`

3. **"Database connection failed"**
   - Verify database is running
   - Check configuration values
   - Ensure network connectivity

4. **"No pending migrations"**
   - Check if migrations directory exists
   - Verify migration files are valid

### Debug Mode
Enable debug logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Schema Validation**: Add schema validation tools
2. **Migration Testing**: Automated migration testing in CI/CD
3. **Performance Monitoring**: Track migration performance
4. **Multi-Environment Support**: Enhanced environment-specific configurations
