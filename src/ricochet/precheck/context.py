from __future__ import annotations

from ..canonical import alarms_from_records
from ..maintenance import MaintenanceStore
from .enrichment import enrichment_completeness
from .similar import HistoricalIncidents
from .runbooks import RunbookRegistry
from .risk import assess_risk


def _key_of(alarm: dict) -> tuple:
    return (alarm.get("integration", ""), alarm.get("system", ""), alarm.get("alert_group", ""))


def _side_match(pair_key: tuple, system: str, alert_group: str) -> bool:
    return pair_key[1] == system and pair_key[2] == alert_group


def root_cause_candidates(alarm_keys: list[tuple], pairs: list[dict]) -> list[dict]:
    if not pairs:
        return []
    groups = {(k[1], k[2]) for k in alarm_keys}
    out = []
    for p in pairs:
        ka, kb = p["key_a"], p["key_b"]
        a_in = any(_side_match(ka, s, g) for s, g in groups)
        b_in = any(_side_match(kb, s, g) for s, g in groups)
        if not (a_in or b_in):
            continue
        lag = p.get("lag_bins", 0)
        leader, follower = (ka, kb) if lag >= 0 else (kb, ka)
        out.append(dict(
            system=leader[1],
            integration=leader[0],
            alert_group=leader[2],
            in_incident=bool(a_in if leader is ka else b_in),
            predicts_downstream=follower[2],
            corr=p.get("corr"),
            lead=p.get("lead"),
            lag_min=p.get("lag_min"),
            days_ok=p.get("days_ok"),
            days_total=p.get("days_total"),
            regions_ok=p.get("regions_ok"),
            class_=p.get("class"),
            month_stable=p.get("month_stable"),
            score=p.get("score"),
        ))
    out.sort(key=lambda c: -(c.get("score") or 0))
    seen, uniq = set(), []
    for c in out:
        k = (c["system"], c["alert_group"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq[:5]


def assemble_case(incident: dict, *, alarms_history, pairs: list[dict],
                   maintenance: MaintenanceStore, incidents: HistoricalIncidents,
                   runbooks: RunbookRegistry) -> dict:
    records = incident.get("alarms", [])
    alarms = alarms_from_records(records)
    keys = [_key_of(r) for r in records]
    profile = {(k[1], k[2]) for k in keys}

    kept, dropped = maintenance.filter_alarms(alarms)
    windows_hit = []
    for r in records:
        for w in maintenance.hit(r["ts"], r.get("region", ""), r.get("system", ""), r.get("node", "")):
            windows_hit.append(w.to_dict())
    seen_w, uniq_windows = set(), []
    for w in windows_hit:
        if w["id"] not in seen_w:
            seen_w.add(w["id"])
            uniq_windows.append(w)

    enrichment = enrichment_completeness(alarms)

    similar = incidents.search(profile, region=incident.get("region", ""))

    top_group = max(set(zip(alarms.get("system", []), alarms.get("alert_group", []))),
                    key=lambda sg: list(zip(alarms["system"], alarms["alert_group"])).count(sg),
                    default=None) if not alarms.empty else None
    runbook = runbooks.match(*top_group) if top_group else None

    candidates = root_cause_candidates(keys, pairs)

    n_nodes = alarms["node"].nunique() if not alarms.empty else 0
    n_regions = alarms["region"].nunique() if not alarms.empty else 0
    max_sev = int(alarms["severity"].max()) if not alarms.empty else 0

    flags_raw = []
    if uniq_windows:
        flags_raw.append(("MAINTENANCE_OVERLAP",
                           f"{len(dropped)} alarms inside {len(uniq_windows)} window(s)"))
    if candidates:
        flags_raw.append(("BUTTERFLY_LINKED",
                          f"{len(candidates)} hidden-dependency candidate(s)"))
    if enrichment["completeness"] < 0.6:
        flags_raw.append(("LOW_ENRICHMENT",
                          f"completeness {enrichment['completeness']:.2f}"))
    if n_nodes >= 10 or n_regions >= 3:
        flags_raw.append(("BLAST_RADIUS", f"{n_nodes} nodes, {n_regions} regions"))
    if max_sev >= 4:
        flags_raw.append(("HIGH_SEVERITY", f"max severity {max_sev}"))
    strong_similar = [s for s in similar if s["similarity"] >= 0.5]
    if len(strong_similar) >= 3:
        flags_raw.append(("REPEAT_OFFENDER", f"{len(strong_similar)} similar incidents"))

    case = dict(
        incident_id=incident.get("incident_id") or incident.get("id") or "",
        received_at=incident.get("received_at"),
        alarms_total=len(records),
        alarms_outside_maintenance=len(kept),
        maintenance=dict(hit=bool(uniq_windows), windows=uniq_windows),
        enrichment=enrichment,
        root_cause_candidates=candidates,
        similar_incidents=similar,
        runbook=runbook,
        blast=dict(nodes=n_nodes, regions=n_regions, max_severity=max_sev),
        risk_flags_raw=flags_raw,
    )
    case["risk"] = assess_risk(case)
    del case["risk_flags_raw"]
    return case
