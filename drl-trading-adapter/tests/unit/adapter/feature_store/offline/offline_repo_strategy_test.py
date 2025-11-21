"""
Unit tests for OfflineRepoStrategy.

These tests verify that the strategy correctly creates the appropriate
repository implementation based on configuration settings.
"""
import pytest
from unittest.mock import patch, MagicMock

from drl_trading_adapter.adapter.feature_store.offline.offline_repo_strategy import OfflineRepoStrategy
from drl_trading_common.config.feature_config import FeatureStoreConfig, LocalRepoConfig, S3RepoConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum


class TestOfflineRepoStrategy:
    """Unit tests for OfflineRepoStrategy."""

    def test_create_local_repository_with_new_config(self) -> None:
        """Test creating local repository using new configuration structure."""
        # Given
        local_config = LocalRepoConfig(repo_path="/test/local/path")
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL,
            local_repo_config=local_config
        )

        strategy = OfflineRepoStrategy(config)

        # When
        with patch('drl_trading_adapter.adapter.feature_store.offline.offline_repo_strategy.OfflineLocalParquetFeatureRepo') as mock_local_repo:
            mock_instance = MagicMock()
            mock_local_repo.return_value = mock_instance

            result = strategy.create_offline_repository()

            # Then
            assert result == mock_instance
            mock_local_repo.assert_called_once()

            # Verify the config passed to the repository has the correct structure
            call_args = mock_local_repo.call_args[0][0]
            assert isinstance(call_args, FeatureStoreConfig)
            assert call_args.local_repo_config is not None
            assert call_args.local_repo_config.repo_path == "/test/local/path"

    def test_create_s3_repository_with_new_config(self) -> None:
        """Test creating S3 repository using new configuration structure."""
        # Given
        s3_config = S3RepoConfig(
            bucket_name="test-bucket",
            prefix="test-prefix",
            region="eu-west-1",
            endpoint_url="http://localhost:9000",
            access_key_id="test-key",
            secret_access_key="test-secret"
        )
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.S3,
            s3_repo_config=s3_config
        )

        strategy = OfflineRepoStrategy(config)

        # When
        with patch('drl_trading_adapter.adapter.feature_store.offline.offline_repo_strategy.OfflineS3ParquetFeatureRepo') as mock_s3_repo:
            mock_instance = MagicMock()
            mock_s3_repo.return_value = mock_instance

            result = strategy.create_offline_repository()

            # Then
            assert result == mock_instance
            mock_s3_repo.assert_called_once()

            # Verify the config passed to the repository has the correct structure
            call_args = mock_s3_repo.call_args[0][0]
            assert isinstance(call_args, FeatureStoreConfig)
            assert call_args.s3_repo_config is not None
            assert call_args.s3_repo_config.bucket_name == "test-bucket"
            assert call_args.s3_repo_config.prefix == "test-prefix"
            assert call_args.s3_repo_config.region == "eu-west-1"
            assert call_args.s3_repo_config.endpoint_url == "http://localhost:9000"
            assert call_args.s3_repo_config.access_key_id == "test-key"
            assert call_args.s3_repo_config.secret_access_key == "test-secret"

    def test_unsupported_strategy_raises_error(self) -> None:
        """Test that unsupported strategy raises ValueError."""
        # Given - Create config with valid enum but test runtime error handling
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL
        )

        strategy = OfflineRepoStrategy(config)

        # Mock the strategy to return an invalid value at runtime
        strategy.config.offline_repo_strategy = "UNSUPPORTED"  # type: ignore

        # When/Then
        with pytest.raises(ValueError, match="Unsupported offline repo strategy"):
            strategy.create_offline_repository()

    def test_local_repository_missing_config_raises_error(self) -> None:
        """Test that missing local configuration raises ValueError."""
        # Given
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.LOCAL
            # No local_repo_config provided
        )

        strategy = OfflineRepoStrategy(config)

        # When/Then
        with pytest.raises(ValueError, match="local_repo_config required for LOCAL strategy"):
            strategy.create_offline_repository()

    def test_s3_repository_missing_config_raises_error(self) -> None:
        """Test that missing S3 configuration raises ValueError."""
        # Given
        config = FeatureStoreConfig(
            cache_enabled=True,
            config_directory="/test/config",
            entity_name="test_entity",
            ttl_days=30,
            service_name="test_service",
            service_version="1.0.0",
            offline_repo_strategy=OfflineRepoStrategyEnum.S3
            # No s3_repo_config provided
        )

        strategy = OfflineRepoStrategy(config)

        # When/Then
        with pytest.raises(ValueError, match="s3_repo_config is required for S3 offline repository strategy"):
            strategy.create_offline_repository()
