"""
Orchestrators coordinate multiple services to achieve complex business workflows.

Orchestrators differ from services in that they:
- Coordinate multiple specialized services
- Manage multi-step workflows and decision trees
- Handle cross-cutting concerns (error handling, notifications)
- Delegate actual work to domain services

This package contains orchestrators that implement business use cases
by composing services, ports, and domain logic.
"""

from drl_trading_preprocess.core.orchestrator.preprocessing_orchestrator import PreprocessingOrchestrator

__all__ = ["PreprocessingOrchestrator"]
