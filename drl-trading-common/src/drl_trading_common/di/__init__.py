"""Modern dependency injection system for DRL Trading.

This package provides a clean, annotation-based dependency injection system
using the injector library, replacing the verbose dependency-injector approach.
"""

# Injectable bootstrap classes
from .injectable_bootstrap import (
    BaseBootstrap,
    DataIngestionBootstrap,
    ExecutionBootstrap,
    InferenceBootstrap,
    TrainingBootstrap,
    get_data_ingestion_bootstrap,
    get_execution_bootstrap,
    get_inference_bootstrap,
    get_training_bootstrap,
)

# Injectable services
from .injectable_services import (
    AgentTrainingService,
    ContextFeatureService,
    DataImportManager,
    DataImportStrategyFactory,
    FeastService,
    FeatureAggregator,
    MergeService,
    PreprocessService,
    RealTimeFeatureAggregator,
    RealTimePreprocessService,
    SplitService,
    StripService,
    get_data_import_manager,
    get_preprocess_service,
    get_real_time_preprocess_service,
    get_split_service,
    get_training_service,
)

# Core DI system
from .injector_container import (
    ApplicationModule,
    TradingInjector,
    get_service,
    inject_dependencies,
)

# Modern container for backward compatibility
from .modern_container import (
    ModernApplicationContainer,
    application_container,
    create_application_container,
)

__all__ = [
    # Core DI
    "TradingInjector",
    "ApplicationModule",
    "get_service",
    "inject_dependencies",
    # Injectable services
    "MergeService",
    "StripService",
    "ContextFeatureService",
    "SplitService",
    "DataImportStrategyFactory",
    "DataImportManager",
    "FeastService",
    "FeatureAggregator",
    "RealTimeFeatureAggregator",
    "PreprocessService",
    "RealTimePreprocessService",
    "AgentTrainingService",
    "get_preprocess_service",
    "get_training_service",
    "get_real_time_preprocess_service",
    "get_data_import_manager",
    "get_split_service",
    # Modern container
    "ModernApplicationContainer",
    "create_application_container",
    "application_container",
    # Injectable bootstrap classes
    "BaseBootstrap",
    "TrainingBootstrap",
    "InferenceBootstrap",
    "DataIngestionBootstrap",
    "ExecutionBootstrap",
    "get_training_bootstrap",
    "get_inference_bootstrap",
    "get_data_ingestion_bootstrap",
    "get_execution_bootstrap",
]
