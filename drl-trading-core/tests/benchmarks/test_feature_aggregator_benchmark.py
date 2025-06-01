import logging

logger = logging.getLogger(__name__)


# class TestFeatureAggregatorBenchmark:
#     """Benchmark test class for feature computation vs. cache retrieval performance.

#     This class provides benchmarks to compare the performance of:
#     1. Computing features from scratch (cold cache)
#     2. Retrieving features from the feature store cache (warm cache)

#     It ensures proper isolation between tests by controlling the feature store state.
#     """

#     @pytest.fixture
#     def dataset(self, mocked_container: ApplicationContainer) -> AssetPriceDataSet:
#         """Get a test dataset using the test container.

#         Returns:
#             A test dataset (H1 timeframe) for benchmarking
#         """
#         # Given
#         importer = mocked_container.data_import_manager()

#         # Get all symbol containers
#         symbol_containers = importer.get_data()

#         # Extract datasets from all symbols
#         all_datasets = []
#         for symbol_container in symbol_containers:
#             all_datasets.extend(symbol_container.datasets)

#         # Filter to get H1 timeframe datasets
#         h1_datasets = [dataset for dataset in all_datasets if dataset.timeframe == "H1"]
#         assert len(h1_datasets) > 0, "No H1 datasets found for benchmarking"
#         return h1_datasets[0]

#     @pytest.fixture
#     def clean_feature_store(self, mocked_container: ApplicationContainer):
#         """Ensure the feature store is empty and properly initialized.

#         This fixture completely cleans the feature store and reinitializes it,
#         ensuring no cached data remains from previous runs.

#         Returns:
#             Function that will clean the feature store when called
#         """

#         def _clean():
#             config = mocked_container.application_config()
#             repo_path = config.feature_store_config.repo_path
#             store_path = config.feature_store_config.offline_store_path

#             # Clean registry DB
#             registry_db = os.path.join(repo_path, "data", "registry.db")
#             if os.path.exists(registry_db):
#                 os.remove(registry_db)
#                 logger.info(f"Removed feature registry DB: {registry_db}")

#             # Clean store data
#             if os.path.exists(store_path):
#                 shutil.rmtree(store_path)
#                 os.makedirs(store_path, exist_ok=True)
#                 logger.info(f"Cleaned feature store data directory: {store_path}")

#             # Initialize repository
#             try:
#                 if not os.path.exists(os.path.join(repo_path, "data")):
#                     os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)

#                 subprocess.run(
#                     ["feast", "apply"], cwd=repo_path, check=True, capture_output=True
#                 )
#                 logger.info("Feature store initialized for benchmarking")
#             except Exception as e:
#                 logger.error(f"Error setting up feast repository: {e}")
#                 raise

#         # Clean before yielding
#         _clean()

#         yield _clean

#     @pytest.fixture
#     def populate_feature_store(
#         self, mocked_container: ApplicationContainer, dataset, clean_feature_store
#     ):
#         """Populate the feature store with computed features.

#         This fixture ensures the feature store is populated with features
#         before testing cache retrieval performance. It first cleans the store,
#         then computes all features.

#         Returns:
#             The symbol used for feature computation
#         """
#         # Given - Get services and configs
#         feature_aggregator = mocked_container.feature_aggregator()
#         feast_service = mocked_container.feast_service()
#         symbol = "EURUSD"  # From test config

#         # Ensure feature store is enabled
#         assert (
#             feast_service.is_enabled()
#         ), "Feature store must be enabled for benchmark tests"

#         # When - Compute all features to populate store
#         tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
#         dask.compute(*tasks)

#         # Wait a moment to ensure all disk writes complete
#         import time

#         time.sleep(1)

#         logger.info("Feature store populated with computed features")
#         return symbol

#     def test_benchmark_compute_features_cold_cache(
#         self,
#         benchmark: BenchmarkFixture,
#         mocked_container,
#         dataset,
#         clean_feature_store,
#     ):
#         """Benchmark feature computation with a cold cache (no cached features).

#         Given: A clean feature store with no cached features
#         When: All features are computed for the dataset
#         Then: Benchmark the time taken for full computation
#         """
#         # Given
#         feature_aggregator: FeatureAggregatorInterface = (
#             mocked_container.feature_aggregator()
#         )
#         symbol = "EURUSD"

#         # When & Then
#         def compute_features():
#             """Compute all features for the dataset."""
#             tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
#             results = dask.compute(*tasks)
#             # Make sure we actually got results
#             assert any(df is not None for df in results), "No features were computed"

#         # Benchmark the computation
#         benchmark.pedantic(compute_features, rounds=3, iterations=1, warmup_rounds=0)

#     def test_benchmark_retrieve_features_warm_cache(
#         self,
#         benchmark: BenchmarkFixture,
#         mocked_container,
#         dataset,
#         populate_feature_store,
#     ):
#         """Benchmark feature retrieval with a warm cache (all features pre-computed).

#         Given: A feature store with all features pre-computed
#         When: All features are requested for the dataset
#         Then: Benchmark the time taken for cache retrieval
#         """
#         # Given
#         feature_aggregator: FeatureAggregatorInterface = (
#             mocked_container.feature_aggregator()
#         )
#         symbol = populate_feature_store  # Symbol returned by fixture

#         # When & Then
#         def retrieve_features():
#             """Retrieve all features for the dataset."""
#             tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
#             results = dask.compute(*tasks)
#             # Make sure we actually got results
#             assert any(df is not None for df in results), "No features were retrieved"

#         # Benchmark the retrieval with warm cache
#         benchmark.pedantic(
#             retrieve_features,
#             rounds=5,  # More rounds for caching as it should be faster
#             iterations=1,
#             warmup_rounds=1,  # 1 warmup to ensure cache is hot
#         )

#     def test_verify_computation_vs_cache_retrieval(
#         self, mocked_container, dataset, clean_feature_store
#     ):
#         """Verify that computation and cache retrieval produce identical results.

#         This test ensures the benchmark is comparing equivalent operations by
#         verifying that feature computation and cache retrieval produce
#         identical DataFrame results.

#         Given: A clean feature store
#         When: Features are computed and then retrieved from cache
#         Then: Both operations should produce identical results
#         """
#         # Given
#         feature_aggregator: FeatureAggregatorInterface = (
#             mocked_container.feature_aggregator()
#         )
#         symbol = "EURUSD"

#         # When - First compute features
#         first_run_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
#         first_run_results = dask.compute(*first_run_tasks)
#         first_run_dfs = [df for df in first_run_results if df is not None]

#         # Create a merged result for validation
#         first_run_merged = pd.DataFrame({"Time": dataset.asset_price_dataset["Time"]})
#         for df in first_run_dfs:
#             if len(df.columns) > 1:  # Only merge if it has features beyond Time
#                 first_run_merged = pd.merge(first_run_merged, df, on="Time", how="left")

#         # Second run - should use cache
#         second_run_tasks = feature_aggregator.compute(asset_data=dataset, symbol=symbol)
#         second_run_results = dask.compute(*second_run_tasks)
#         second_run_dfs = [df for df in second_run_results if df is not None]

#         # Create a merged result for validation
#         second_run_merged = pd.DataFrame({"Time": dataset.asset_price_dataset["Time"]})
#         for df in second_run_dfs:
#             if len(df.columns) > 1:
#                 second_run_merged = pd.merge(
#                     second_run_merged, df, on="Time", how="left"
#                 )

#         # Then - Verify results match
#         assert len(first_run_dfs) == len(
#             second_run_dfs
#         ), "Number of result DataFrames should match"

#         # Sort columns to ensure consistent order for comparison
#         first_cols = sorted(first_run_merged.columns)
#         second_cols = sorted(second_run_merged.columns)

#         assert first_cols == second_cols, "Column names should match between runs"

#         # Compare actual data values
#         pd.testing.assert_frame_equal(
#             first_run_merged[first_cols],
#             second_run_merged[second_cols],
#             check_dtype=False,  # Allow for small type differences that don't affect values
#         )
#         logger.info(
#             "Verified: Computation and cache retrieval produce identical results"
#         )
