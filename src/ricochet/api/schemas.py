from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AlarmIn(BaseModel):
    ts: datetime
    integration: str = ""
    system: str = ""
    alert_group: str = ""
    node: str = ""
    node_alias: str = ""
    region: str = ""
    severity: int = 4


class IncidentIn(BaseModel):
    incident_id: str = ""
    received_at: Optional[datetime] = None
    region: str = ""
    alarms: list[AlarmIn]


class RunParams(BaseModel):
    bin_min: Optional[int] = None
    min_events: Optional[int] = None
    min_bins: Optional[int] = None
    lags: Optional[int] = None
    min_abs_corr: Optional[float] = None
    min_support: Optional[int] = None
    hub_degree: Optional[int] = None
    max_duty: Optional[float] = None
    min_days_frac: Optional[float] = None
    min_hours_frac: Optional[float] = None
    transform: Optional[str] = None
    system_field: Optional[str] = None
    drop_region: Optional[bool] = None
    region_min_abs: Optional[float] = None
    min_lead: Optional[float] = None
    min_devices: Optional[int] = None
    remove_common_factor: Optional[bool] = None
    holdout_last: int = 1
    ts_from: Optional[datetime] = None
    ts_to: Optional[datetime] = None


class SeriesKeyIn(BaseModel):
    integration: str
    system: str
    alert_group: str


class VerifyPairIn(BaseModel):
    key_a: SeriesKeyIn
    key_b: SeriesKeyIn
    lag_bins: int = 0


class WindowIn(BaseModel):
    start: datetime
    end: datetime
    region: str = ""
    system: str = ""
    node: str = ""
    source: str = "api"
    note: str = ""


class RunbookIn(BaseModel):
    id: str
    system: str = ""
    alert_group: str = ""
    url: str = ""
    steps: list[str] = Field(default_factory=list)
    owner: str = ""
