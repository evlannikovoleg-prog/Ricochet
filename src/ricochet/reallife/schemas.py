from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class RealtimeAlarm(BaseModel):
    ts: datetime
    integration: str = ""
    system: str = ""
    alert_group: str = ""
    node: str = ""
    node_alias: str = ""
    region: str = ""
    severity: int = 4
    event_id: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopologyNodeIn(BaseModel):
    id: str = ""
    node: str = ""
    node_alias: str = ""
    integration: str = ""
    system: str = ""
    region: str = ""
    device_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopologyEdgeIn(BaseModel):
    source: str
    target: str
    relation: str = ""
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MaintenanceWindowIn(BaseModel):
    start: datetime
    end: datetime
    region: str = ""
    system: str = ""
    node: str = ""
    note: str = ""
    source: str = "realtime"


class IncidentBundleIn(BaseModel):
    incident_id: str = ""
    received_at: Optional[datetime] = None
    region: str = ""
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    alarms: list[RealtimeAlarm] = Field(default_factory=list)
    topology_nodes: list[TopologyNodeIn] = Field(default_factory=list)
    topology_edges: list[TopologyEdgeIn] = Field(default_factory=list)
    maintenance: list[MaintenanceWindowIn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
