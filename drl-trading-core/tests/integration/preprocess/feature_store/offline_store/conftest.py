"""
TestContainer fixtures for S3-based offline feature storage testing.

Provides LocalStack/MinIO containers for realistic S3 testing without needing AWS credentials.
"""

from typing import Generator

import boto3
import pandas as pd
import pytest
from drl_trading_common.config.feature_config import FeatureStoreConfig, S3RepoConfig
from drl_trading_common.enum.offline_repo_strategy_enum import OfflineRepoStrategyEnum
from pandas import DataFrame
from testcontainers.minio import MinioContainer


@pytest.fixture(scope="session")
def minio_container() -> Generator[MinioContainer, None, None]:
    """
    Start a MinIO container for S3-compatible testing.

    MinIO is lighter and faster than LocalStack for pure S3 testing.
    """
    with MinioContainer() as minio:
        yield minio


@pytest.fixture
def s3_client_minio(minio_container: MinioContainer) -> boto3.client:
    """Create a boto3 S3 client connected to MinIO container."""
    return boto3.client(
        "s3",
        endpoint_url=f"http://{minio_container.get_container_host_ip()}:{minio_container.get_exposed_port(9000)}",
        aws_access_key_id=minio_container.access_key,
        aws_secret_access_key=minio_container.secret_key,
        region_name="us-east-1"
    )

@pytest.fixture
def s3_test_bucket(s3_client_minio: boto3.client) -> str:
    """Create a test bucket for feature storage testing."""
    import uuid
    # Use a unique bucket name per test session to ensure isolation
    bucket_name = f"test-feature-store-{uuid.uuid4().hex[:8]}"

    # Try to create bucket, ignore if it already exists
    try:
        s3_client_minio.create_bucket(Bucket=bucket_name)
    except s3_client_minio.exceptions.BucketAlreadyOwnedByYou:
        # Bucket already exists, which is fine for our tests
        pass
    except s3_client_minio.exceptions.BucketAlreadyExists:
        # Bucket exists but owned by someone else (shouldn't happen in MinIO)
        pass

    return bucket_name


@pytest.fixture
def s3_feature_store_config(
    s3_client_minio: boto3.client,
    s3_test_bucket: str,
    minio_container: MinioContainer
) -> FeatureStoreConfig:
    """Create FeatureStoreConfig for S3 testing."""
    s3_repo_config = S3RepoConfig(
        endpoint_url=f"http://{minio_container.get_container_host_ip()}:{minio_container.get_exposed_port(9000)}",
        bucket_name=s3_test_bucket,
        access_key_id=minio_container.access_key,
        secret_access_key=minio_container.secret_key,
        region="us-east-1"
    )

    return FeatureStoreConfig(
        enabled=True,
        config_directory="/tmp/test_repo",  # Required field for new structure
        entity_name="test_entity",
        ttl_days=30,
        online_enabled=False,
        service_name="test_service",
        service_version="1.0.0",
        offline_repo_strategy=OfflineRepoStrategyEnum.S3,
        s3_repo_config=s3_repo_config
    )


@pytest.fixture
def offline_s3_repo(s3_feature_store_config: FeatureStoreConfig):
    """Create OfflineFeatureS3Repo for testing with real S3 backend."""
    from drl_trading_core.preprocess.feature_store.offline_store.offline_feature_s3_repo import (
        OfflineFeatureS3Repo,
    )

    return OfflineFeatureS3Repo(s3_feature_store_config)


@pytest.fixture
def sample_features_df() -> DataFrame:
    """Create sample features DataFrame with realistic data."""
    return DataFrame({
        "event_timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 11:00:00")
        ],
        "feature_1": [1.5, 2.5, 3.5],
        "feature_2": [10.0, 20.0, 30.0],
        "rsi_14": [30.5, 45.2, 67.8]
    })


@pytest.fixture
def eurusd_h1_symbol() -> str:
    """Create a sample symbol for EUR/USD H1 with unique identifier per test."""
    import uuid
    return f"EURUSD-{uuid.uuid4().hex[:8]}"
