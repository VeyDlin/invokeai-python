# Path: invoke\api\queue\schema.py
from typing import List, Any, Optional
from pydantic import BaseModel


class Graph(BaseModel):
    id: str
    nodes: Any
    edges: Any


class Batch(BaseModel):
    batch_id: str
    origin: Optional[str]
    destination: Optional[str]
    data: Optional[List[List[Any]]]
    graph: Graph
    workflow: Optional[Any]
    runs: int
    

class BatchStatus(BaseModel):
    queue_id: str
    batch_id: str
    pending: int
    in_progress: int
    completed: int
    failed: int
    canceled: int
    total: int


class FieldValue(BaseModel):
    node_path: str
    field_name: str
    value: str


class Item(BaseModel):
    item_id: int
    status: str
    priority: int
    batch_id: str
    origin: Optional[str]
    destination: Optional[str]
    session_id: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    error_traceback: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    queue_id: str
    field_values: Optional[List[FieldValue]]


class CursorPaginatedResults(BaseModel):
    limit: int
    has_more: bool
    items: List[Item]


class EnqueueBatch(BaseModel):
    queue_id: str
    enqueued: int
    requested: int
    batch: Batch
    priority: int


class Session(BaseModel):
    id: str
    graph: Graph
    execution_graph: Optional[Graph]
    executed: Optional[List[str]]
    executed_history: Optional[List[str]]
    results: Optional[Any]
    errors: Optional[Any]
    prepared_source_mapping: Optional[Any]
    source_prepared_mapping: Optional[Any]


class SessionQueueItem(BaseModel):
    item_id: int
    status: str
    priority: int
    batch_id: str
    origin: Optional[str]
    destination: Optional[str]
    session_id: str
    error_type: Optional[str]
    error_message: Optional[str]
    error_traceback: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    queue_id: str
    field_values: Optional[List[FieldValue]]
    session: Session
    workflow: Optional[Any]


class QueueStatus(BaseModel):
    queue_id: str
    item_id: Optional[int]
    batch_id: Optional[str]
    session_id: Optional[str]
    pending: int
    in_progress: int
    completed: int
    failed: int
    canceled: int
    total: int


class ProcessorStatus(BaseModel):
    is_started: bool
    is_processing: bool


class QueueProcessorStatus(BaseModel):
    queue: QueueStatus
    processor: ProcessorStatus


class ProcessorResponse(BaseModel):
    is_started: bool
    is_processing: bool