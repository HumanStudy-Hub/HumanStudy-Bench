"""Source connector backbone for fetching paper supplementary materials."""

from generation_pipeline.connectors.base_connector import (
    BaseSourceConnector,
    ConnectorFetchResult,
    FetchedSource,
    SourceFetchPlan,
)
from generation_pipeline.connectors.registry import SourceConnectorRegistry

__all__ = [
    "BaseSourceConnector",
    "ConnectorFetchResult",
    "FetchedSource",
    "SourceFetchPlan",
    "SourceConnectorRegistry",
]
