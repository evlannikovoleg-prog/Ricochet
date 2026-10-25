from __future__ import annotations

import json
from pathlib import Path

REQUIRED_FIELDS = ["node", "node_alias", "region", "alert_group", "system", "integration"]


def enrichment_completeness(alarms) -> dict:
    if len(alarms) == 0:
        return dict(completeness=0.0, per_field={}, missing_all=True)
    per_field = {}
    for f in REQUIRED_FIELDS:
        filled = (alarms[f].fillna("").astype(str).str.len() > 0).mean()
        per_field[f] = round(float(filled), 4)
    completeness = sum(per_field.values()) / len(REQUIRED_FIELDS)
    return dict(
        completeness=round(float(completeness), 4),
        per_field=per_field,
        missing_all=False,
    )
