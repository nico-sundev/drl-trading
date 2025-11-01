"""Factory for creating Kafka message handlers."""

import logging
from confluent_kafka import Message

from drl_trading_common.messaging.kafka_handler_decorator import kafka_handler
from drl_trading_common.messaging.kafka_message_handler import KafkaMessageHandler
from drl_trading_common.model.feature_preprocessing_request import FeaturePreprocessingRequest
from drl_trading_preprocess.adapter.messaging.kafka_handler_constants import HANDLER_ID_PREPROCESSING_REQUEST
from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator


logger = logging.getLogger(__name__)


class KafkaMessageHandlerFactory:
    """Factory for creating Kafka message handlers."""

    @staticmethod
    def create_preprocessing_request_handler(orchestrator: PreprocessingOrchestrator) -> KafkaMessageHandler:
        """
        Create handler for preprocessing request messages.
        
        Args:
            orchestrator: Preprocessing orchestrator to invoke
            
        Returns:
            Handler function that parses FeaturePreprocessingRequest and invokes orchestrator
        """
        @kafka_handler(FeaturePreprocessingRequest, HANDLER_ID_PREPROCESSING_REQUEST)
        def handler(request: FeaturePreprocessingRequest, message: Message) -> None:
            orchestrator.process_feature_computation_request(request)
        
        return handler
