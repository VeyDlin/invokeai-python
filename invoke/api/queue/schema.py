# Path: invoke\api\queue\schema.py
from typing import List, Any, Optional
from pydantic import BaseModel


class Graph(BaseModel):
    id: str
    nodes: Any
    edges: List["Edge"]


class Batch(BaseModel):
    batch_id: str
    graph: Graph
    runs: int
    data: List[List[Any]]


class BatchStatus(BaseModel):
    queue_id: str
    batch_id: str
    pending: int
    in_progress: int
    completed: int
    failed: int
    canceled: int
    total: int


class Clear(BaseModel):
    deleted: int


class FieldValue(BaseModel):
    node_path: str
    field_name: str
    value: str


class Item(BaseModel):
    item_id: int
    status: str
    priority: int
    batch_id: str
    session_id: str
    error: str
    created_at: str
    updated_at: str
    started_at: str
    completed_at: str
    queue_id: str
    field_values: List[FieldValue]


class CursorPaginatedResults(BaseModel):
    limit: int
    has_more: bool
    items: List[Item]


class EdgePoint(BaseModel):
    node_id: str
    field: str


class Edge(BaseModel):
    source: EdgePoint
    destination: EdgePoint


class EnqueueBatch(BaseModel):
    queue_id: str
    enqueued: int
    requested: int
    batch: Batch
    priority: int


class Session(BaseModel):
    id: str
    graph: Graph
    execution_graph: Graph
    executed: List[str]
    executed_history: List[str]
    results: Any
    errors: Any
    prepared_source_mapping: Any
    source_prepared_mapping: Any


class SessionQueueItem(BaseModel):
    item_id: int
    status: str
    priority: int
    batch_id: str
    session_id: str
    error: str
    created_at: str
    updated_at: str
    started_at: str
    completed_at: str
    queue_id: str
    field_values: List[Any]
    session: Session


class QueueStatus(BaseModel):
    active: bool
    pending: int
    completed: int
    errored: int
    total: int


class ProcessorResponse(BaseModel):
    status: str 
    message: Optional[str]


class CancelResponse(BaseModel):
    cancelled: int
    destination: str


class CountsByDestinationResponse(BaseModel):
    destination: str
    count: int


class PruneResponse(BaseModel):
    success: bool
    message: Optional[str]


class CancelByBatchIdsResponse(BaseModel):
    cancelled: int 
    batch_ids: List[str]


class CancelQueueItemResponse(BaseModel):
    cancelled: bool 
    item_id: str