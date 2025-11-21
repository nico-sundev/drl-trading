"""
S3 implementation of offline feature repository.

This module provides feature storage and retrieval using S3 buckets
with the same datetime-based organization as the local filesystem implementation.
"""

import json
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from drl_trading_common.config.feature_config import FeatureStoreConfig
from injector import inject
from pandas import DataFrame

from drl_trading_adapter.adapter.feature_store.offline.parquet.base_parquet_feature_repo import BaseParquetFeatureRepo

logger = logging.getLogger(__name__)


class S3StorageException(Exception):
    """Exception raised for S3 storage operations."""
    pass


@inject
class OfflineS3ParquetFeatureRepo(BaseParquetFeatureRepo):
    """
    S3 implementation for offline feature storage.

    Features are stored as parquet files in S3 buckets using the same
    datetime-based organization as the local filesystem implementation:
    s3://bucket/symbol/year=YYYY/month=MM/day=DD/features_*.parquet

    This implementation provides:
    - Scalable cloud storage
    - Same API as local filesystem
    - Efficient temporal queries via S3 prefix filtering
    - Cost-effective storage with S3 lifecycle policies
    """

    def __init__(self, config: FeatureStoreConfig):
        """
        Initialize S3 repository with configuration.

        Args:
            config: Feature store configuration with S3 settings

        Raises:
            S3StorageException: If S3 configuration is invalid or credentials missing
        """
        if config.offline_repo_strategy.value != "s3":
            raise ValueError(f"OfflineS3ParquetFeatureRepo requires S3 strategy, got {config.offline_repo_strategy}")

        if not config.s3_repo_config:
            raise ValueError("s3_repo_config is required for S3 offline repository strategy")

        self.config = config
        self.bucket_name = config.s3_repo_config.bucket_name
        self.s3_prefix = config.s3_repo_config.prefix

        # Initialize S3 client with proper error handling
        self._s3_client = self._initialize_s3_client()

        # Validate bucket access during initialization
        self._validate_bucket_access()

    # ==================================================================================
    # Implement abstract I/O methods from BaseOfflineFeatureRepo
    # ==================================================================================

    def _load_metadata(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load partition metadata for a symbol from S3."""
        from botocore.exceptions import ClientError

        metadata_key = f"{self.s3_prefix}/{symbol}/_metadata.json"

        try:
            response = self._s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
            metadata_bytes = response["Body"].read()
            return json.loads(metadata_bytes.decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            logger.warning(f"Failed to load metadata for {symbol}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load metadata for {symbol}: {e}")
            return None

    def _save_metadata(self, symbol: str, metadata: Dict[str, Any]) -> None:
        """Save partition metadata for a symbol to S3."""
        metadata_key = f"{self.s3_prefix}/{symbol}/_metadata.json"

        try:
            metadata_bytes = json.dumps(metadata, indent=2).encode('utf-8')
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=metadata_key,
                Body=metadata_bytes,
                ContentType="application/json"
            )
        except Exception as e:
            logger.error(f"Failed to save metadata for {symbol}: {e}")
            raise S3StorageException(f"Failed to save metadata for {symbol}: {e}") from e

    def _load_partitions(self, symbol: str, partitions: List[Dict[str, Any]]) -> DataFrame:
        """Load data from specific S3 partitions only."""
        from botocore.exceptions import ClientError

        dfs = []
        for partition in partitions:
            s3_key = partition["s3_key"]
            try:
                response = self._s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
                parquet_buffer = BytesIO(response['Body'].read())
                df = pd.read_parquet(parquet_buffer, engine="pyarrow")
                dfs.append(df)
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    logger.warning(f"Failed to load partition {s3_key}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load partition {s3_key}: {e}")

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True).sort_values("event_timestamp")

    def _delete_partitions(self, symbol: str, partitions: List[Dict[str, Any]]) -> None:
        """Delete specific S3 partition objects."""
        for partition in partitions:
            s3_key = partition["s3_key"]
            try:
                self._s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
                logger.debug(f"Deleted partition: s3://{self.bucket_name}/{s3_key}")
            except Exception as e:
                logger.error(f"Failed to delete partition {s3_key}: {e}")

    def _store_with_datetime_organization(self, features_df: DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Store features using datetime-based S3 key organization.

        S3 key structure: prefix/symbol/year=YYYY/month=MM/day=DD/features_*.parquet

        Args:
            features_df: Features DataFrame to store
            symbol: Symbol identifier

        Returns:
            List of partition metadata dicts with s3_key, min/max timestamps, and record count
        """
        partition_metadata: List[Dict[str, Any]] = []

        if features_df.empty:
            return partition_metadata

        # Group features by date for organized storage
        features_df = features_df.copy()
        features_df["_date"] = features_df["event_timestamp"].dt.date

        for date_key, group_df in features_df.groupby("_date"):
            # Ensure date_key is a date object
            if hasattr(date_key, 'year'):
                year = date_key.year
                month = date_key.month
                day = date_key.day
            else:
                # Fallback for edge cases
                date_obj = pd.to_datetime(str(date_key)).date()
                year = date_obj.year
                month = date_obj.month
                day = date_obj.day

            # Generate S3 key with datetime organization
            min_time = group_df["event_timestamp"].min()
            max_time = group_df["event_timestamp"].max()
            filename = f"features_{min_time.strftime('%Y%m%d_%H%M%S')}_{max_time.strftime('%H%M%S')}.parquet"

            s3_key = f"{self.s3_prefix}/{symbol}/year={year}/month={month:02d}/day={day:02d}/{filename}"

            # Remove temporary date column before storage
            store_df = group_df.drop(columns=["_date"])

            # Store to S3
            self._store_parquet_to_s3(store_df, s3_key)

            logger.info(
                f"Stored {len(store_df)} features for {year}-{month:02d}-{day:02d} in s3://{self.bucket_name}/{s3_key} "
                f"({symbol})"
            )

            # Track partition metadata
            partition_metadata.append({
                "s3_key": s3_key,
                "min_timestamp": str(min_time),
                "max_timestamp": str(max_time),
                "record_count": len(store_df)
            })

        return partition_metadata

    def get_repo_path(self, symbol: str) -> str:
        if not symbol or not symbol.strip():
            raise ValueError("Symbol cannot be empty or None")

        return f"s3://{self.bucket_name}/{self.s3_prefix}/{symbol}"

    # ==================================================================================
    # S3-specific helper methods
    # ==================================================================================

    def _initialize_s3_client(self) -> boto3.client:
        """Initialize boto3 S3 client with proper configuration."""
        try:
            s3_config = {
                "service_name": "s3",
                "region_name": self.config.s3_repo_config.region
            }

            # Add endpoint URL if specified (for LocalStack/MinIO testing)
            if self.config.s3_repo_config.endpoint_url:
                s3_config["endpoint_url"] = self.config.s3_repo_config.endpoint_url

            # Add credentials if specified
            if self.config.s3_repo_config.access_key_id:
                s3_config["aws_access_key_id"] = self.config.s3_repo_config.access_key_id
                s3_config["aws_secret_access_key"] = self.config.s3_repo_config.secret_access_key

            return boto3.client(**s3_config)

        except Exception as e:
            raise S3StorageException(f"Failed to initialize S3 client: {str(e)}") from e

    def _validate_bucket_access(self) -> None:
        """Validate that the S3 bucket exists and is accessible."""
        try:
            self._s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully validated access to S3 bucket: {self.bucket_name}")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Try to create the bucket
                try:
                    self._s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_e:
                    raise S3StorageException(f"Bucket {self.bucket_name} does not exist and could not be created: {create_e}") from create_e
            else:
                raise S3StorageException(f"Cannot access S3 bucket {self.bucket_name}: {e}") from e
        except NoCredentialsError as e:
            raise S3StorageException(f"No AWS credentials found for S3 access: {e}") from e

    def _list_s3_objects(self, prefix: str, max_keys: Optional[int] = None) -> List[str]:
        """List S3 objects with the given prefix."""
        try:
            kwargs = {
                "Bucket": self.bucket_name,
                "Prefix": prefix
            }
            if max_keys:
                kwargs["MaxKeys"] = max_keys

            response = self._s3_client.list_objects_v2(**kwargs)

            if "Contents" not in response:
                return []

            return [obj["Key"] for obj in response["Contents"] if obj["Key"].endswith('.parquet')]

        except Exception as e:
            raise S3StorageException(f"Failed to list S3 objects with prefix {prefix}: {str(e)}") from e

    def _load_parquet_from_s3(self, s3_key: str) -> DataFrame:
        """Load a parquet file from S3."""
        try:
            response = self._s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            parquet_bytes = response["Body"].read()

            # Use BytesIO to read parquet from bytes
            buffer = BytesIO(parquet_bytes)
            return pd.read_parquet(buffer)

        except Exception as e:
            raise S3StorageException(f"Failed to load parquet from s3://{self.bucket_name}/{s3_key}: {str(e)}") from e

    def _store_parquet_to_s3(self, df: DataFrame, s3_key: str) -> None:
        """Store a DataFrame as parquet to S3."""
        try:
            # Convert DataFrame to parquet bytes
            buffer = BytesIO()
            df.to_parquet(
                buffer,
                index=False,
                compression="snappy",
                engine="pyarrow"
            )
            buffer.seek(0)

            # Upload to S3
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream"
            )

        except Exception as e:
            raise S3StorageException(f"Failed to store parquet to s3://{self.bucket_name}/{s3_key}: {str(e)}") from e
