"""Public data types for the HEBBS Python SDK.

All types are plain dataclasses -- no protobuf leakage in the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryKind(Enum):
    EPISODE = "episode"
    INSIGHT = "insight"
    REVISION = "revision"
    UNSPECIFIED = "unspecified"


class EdgeType(Enum):
    CAUSED_BY = "caused_by"
    RELATED_TO = "related_to"
    FOLLOWED_BY = "followed_by"
    REVISED_FROM = "revised_from"
    INSIGHT_FROM = "insight_from"
    UNSPECIFIED = "unspecified"


class RecallStrategy(Enum):
    SIMILARITY = "similarity"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    ANALOGICAL = "analogical"


@dataclass(frozen=True)
class Edge:
    target_id: bytes
    edge_type: EdgeType
    confidence: float | None = None


@dataclass
class Memory:
    id: bytes
    content: str
    importance: float
    context: dict[str, Any]
    entity_id: str | None = None
    created_at: int = 0
    updated_at: int = 0
    last_accessed_at: int = 0
    access_count: int = 0
    decay_score: float = 0.0
    kind: MemoryKind = MemoryKind.EPISODE
    embedding: list[float] = field(default_factory=list)


@dataclass
class StrategyDetail:
    strategy: str
    relevance: float = 0.0
    distance: float | None = None
    timestamp: int | None = None
    rank: int | None = None
    depth: int | None = None
    embedding_similarity: float | None = None
    structural_similarity: float | None = None


@dataclass
class RecallResult:
    memory: Memory
    score: float
    strategy_details: list[StrategyDetail] = field(default_factory=list)


@dataclass
class StrategyError:
    strategy: str
    message: str


@dataclass
class RecallOutput:
    results: list[RecallResult]
    strategy_errors: list[StrategyError] = field(default_factory=list)


@dataclass
class PrimeOutput:
    results: list[RecallResult]
    temporal_count: int = 0
    similarity_count: int = 0


@dataclass
class ForgetResult:
    forgotten_count: int
    cascade_count: int
    tombstone_count: int
    truncated: bool = False


@dataclass
class ReflectResult:
    insights_created: int
    clusters_found: int
    clusters_processed: int
    memories_processed: int


@dataclass
class SubscribePush:
    subscription_id: int
    memory: Memory
    confidence: float
    push_timestamp_us: int = 0
    sequence_number: int = 0


@dataclass
class HealthStatus:
    serving: bool
    version: str
    memory_count: int
    uptime_seconds: int
