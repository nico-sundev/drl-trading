"""
Mapper for FeaturePreprocessingRequest between adapter and core layers.
"""

from drl_trading_common.adapter.model.feature_preprocessing_request import FeaturePreprocessingRequest as AdapterFeaturePreprocessingRequest
from drl_trading_common.adapter.mappers.feature_config_version_info_mapper import FeatureConfigVersionInfoMapper
from drl_trading_common.adapter.mappers.timeframe_mapper import TimeframeMapper
from drl_trading_core.core.dto.feature_preprocessing_request import FeaturePreprocessingRequest as CoreFeaturePreprocessingRequest


class FeaturePreprocessingRequestMapper:
    """
    Mapper for FeaturePreprocessingRequest between adapter and core layers.

    Handles conversion between the adapter DTO (used for communication) and
    the core domain model (used for business logic).
    """

    @staticmethod
    def dto_to_domain(
        dto: AdapterFeaturePreprocessingRequest
    ) -> CoreFeaturePreprocessingRequest:
        """
        Convert adapter FeaturePreprocessingRequest (DTO) to core FeaturePreprocessingRequest (domain).

        Args:
            dto: FeaturePreprocessingRequest from adapter layer (DTO)

        Returns:
            Corresponding core FeaturePreprocessingRequest domain model
        """
        return CoreFeaturePreprocessingRequest(
            symbol=dto.symbol,
            base_timeframe=TimeframeMapper.dto_to_domain(dto.base_timeframe),
            target_timeframes=[
                TimeframeMapper.dto_to_domain(tf) for tf in dto.target_timeframes
            ],
            feature_config_version_info=FeatureConfigVersionInfoMapper.dto_to_domain(
                dto.feature_config_version_info
            ),
            start_time=dto.start_time,
            end_time=dto.end_time,
            request_id=dto.request_id,
            force_recompute=dto.force_recompute,
            incremental_mode=dto.incremental_mode,
            processing_context=dto.processing_context,
            skip_existing_features=dto.skip_existing_features,
            materialize_online=dto.materialize_online,
            batch_size=dto.batch_size,
            parallel_processing=dto.parallel_processing,
        )

    @staticmethod
    def domain_to_dto(
        domain: CoreFeaturePreprocessingRequest
    ) -> AdapterFeaturePreprocessingRequest:
        """
        Convert core FeaturePreprocessingRequest (domain) to adapter FeaturePreprocessingRequest (DTO).

        Args:
            domain: FeaturePreprocessingRequest from core layer (domain)

        Returns:
            Corresponding adapter FeaturePreprocessingRequest DTO
        """
        return AdapterFeaturePreprocessingRequest(
            symbol=domain.symbol,
            base_timeframe=TimeframeMapper.domain_to_dto(domain.base_timeframe),
            target_timeframes=[
                TimeframeMapper.domain_to_dto(tf) for tf in domain.target_timeframes
            ],
            feature_config_version_info=FeatureConfigVersionInfoMapper.domain_to_dto(
                domain.feature_config_version_info
            ),
            start_time=domain.start_time,
            end_time=domain.end_time,
            request_id=domain.request_id,
            force_recompute=domain.force_recompute,
            incremental_mode=domain.incremental_mode,
            processing_context=domain.processing_context,
            skip_existing_features=domain.skip_existing_features,
            materialize_online=domain.materialize_online,
            batch_size=domain.batch_size,
            parallel_processing=domain.parallel_processing,
        )
