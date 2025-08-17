"""
CLI commands for database migration management.

This module provides command-line interface for managing database migrations
in the DRL Trading Ingest service. It includes commands for applying migrations,
creating new migrations, and checking migration status.
"""

import logging
import os
import sys

import click
from injector import Injector
from drl_trading_common.config.service_config_loader import ServiceConfigLoader
from drl_trading_ingest.infrastructure.config.ingest_config import IngestConfig

from drl_trading_ingest.core.port.migration_service_interface import (
    MigrationError,
    MigrationServiceInterface,
)
from drl_trading_ingest.infrastructure.di.ingest_module import IngestModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_migration_service() -> MigrationServiceInterface:
    """
    Get the migration service instance from the DI container.

    Returns:
        MigrationServiceInterface: Configured migration service
    """
    config: IngestConfig = ServiceConfigLoader.load_config(IngestConfig)
    injector = Injector([IngestModule(config)])
    return injector.get(MigrationServiceInterface)


@click.group()
def cli():
    """Database migration management commands."""
    pass


@cli.command()
def migrate():
    """Apply all pending migrations to the database."""
    try:
        migration_service = get_migration_service()
        migration_service.migrate_to_latest()
        click.echo("✅ All migrations applied successfully")
    except MigrationError as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('revision')
def migrate_to(revision: str):
    """Migrate to a specific revision."""
    try:
        migration_service = get_migration_service()
        migration_service.migrate_to_revision(revision)
        click.echo(f"✅ Successfully migrated to revision: {revision}")
    except MigrationError as e:
        click.echo(f"❌ Migration failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('message')
@click.option('--autogenerate/--no-autogenerate', default=True,
              help='Auto-generate migration from schema changes')
def create_migration(message: str, autogenerate: bool):
    """Create a new migration with the given message."""
    try:
        migration_service = get_migration_service()
        revision_id = migration_service.create_migration(message, autogenerate)
        click.echo(f"✅ Created migration: {revision_id}")
    except MigrationError as e:
        click.echo(f"❌ Migration creation failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show current migration status."""
    try:
        migration_service = get_migration_service()
        current_rev = migration_service.get_current_revision()
        pending = migration_service.get_pending_migrations()

        click.echo("📊 Migration Status:")
        click.echo(f"Current revision: {current_rev or 'None (no migrations applied)'}")

        if pending:
            click.echo(f"Pending migrations: {len(pending)}")
            for rev in pending:
                click.echo(f"  - {rev}")
        else:
            click.echo("✅ No pending migrations")

    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        sys.exit(1)


@cli.command()
def init():
    """Initialize the migration repository."""
    try:
        migration_service = get_migration_service()
        migration_service.initialize_migration_repo()
        click.echo("✅ Migration repository initialized")
    except MigrationError as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    # Ensure SERVICE_CONFIG_PATH is set
    if not os.environ.get("SERVICE_CONFIG_PATH"):
        click.echo("❌ SERVICE_CONFIG_PATH environment variable must be set", err=True)
        sys.exit(1)

    cli()
