from __future__ import annotations


FLAG_WEIGHTS = {
    "MAINTENANCE_OVERLAP": 30,
    "BUTTERFLY_LINKED": 25,
    "LOW_ENRICHMENT": 20,
    "BLAST_RADIUS": 20,
    "HIGH_SEVERITY": 15,
    "REPEAT_OFFENDER": 10,
}


def assess_risk(case: dict) -> dict:
    flags = []
    for f in case.get("risk_flags_raw", []):
        code, detail = f if isinstance(f, tuple) else (f.get("code"), f.get("detail"))
        flags.append(dict(code=code, weight=FLAG_WEIGHTS.get(code, 5), detail=detail or ""))
    score = min(100, sum(f["weight"] for f in flags))
    level = "LOW"
    if score >= 60:
        level = "HIGH"
    elif score >= 30:
        level = "MEDIUM"
    return dict(score=score, level=level, flags=flags)
