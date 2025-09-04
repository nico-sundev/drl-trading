from pandas import DataFrame

from drl_trading_core.core.port import IFeatureStoreFetchPort
from drl_trading_core.preprocess.feature import FeatureManager

from ..port import IFeatureStoreSavePort


class SampleService:
    def __init__(
        self,
        feature_store_save_repo: IFeatureStoreSavePort,
        feature_store_fetch_repo: IFeatureStoreFetchPort,
        feature_manager: FeatureManager,
    ) -> None:
        self.feature_store_save_repo = feature_store_save_repo
        self.feature_store_fetch_repo = feature_store_fetch_repo
        self.feature_manager = feature_manager

    def sample_call(self) -> None:
        self.feature_store_save_repo.store_computed_features_offline(
            DataFrame(), "AAPL", []
        )
