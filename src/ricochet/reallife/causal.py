from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .pipeline import PreparedIncident, series_id
from .schemas import TopologyEdgeIn, TopologyNodeIn


@dataclass
class CausalConfig:
    max_lag_seconds: int = 300
    min_abs_corr: float = 0.35
    min_causal_power: float = 0.30
    max_nodes: int = 30
    max_cond_size: int = 1
    max_cond_candidates: int = 8
    min_active_bins: int = 3
    max_edges: int = 200
    remove_common_factor: bool = True
    use_topology: bool = True
    topology_strict: bool = False


def _zscore(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return np.divide(X - mu, np.where(sd > 0, sd, 1.0))


def _remove_common_factor(X: np.ndarray) -> np.ndarray:
    """Remove the leave-one-out global activity from each series.

    A network-wide storm should not look like a direct causal edge. This is a
    lightweight analogue of the common-factor control used by the batch engine.
    """
    T, N = X.shape
    if N <= 1 or T < 3:
        return X.copy()
    total = X.sum(axis=1)
    out = X.copy()
    for j in range(N):
        other = total - X[:, j]
        if other.std() == 0:
            continue
        A = np.column_stack([np.ones(T), (other - other.mean()) / other.std()])
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        out[:, j] = X[:, j] - A @ beta
    return out


def _residualize(y: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    if Z is None or Z.size == 0:
        return y - y.mean()
    A = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _partial_corr_at_lag(X: np.ndarray, source: int, target: int, lag: int,
                         cond: list[int]) -> float:
    T = X.shape[0]
    x = X[: T - lag, source]
    y = X[lag:, target]
    Z = X[: T - lag, cond] if cond else None
    return _corr(_residualize(x, Z), _residualize(y, Z))


def _best_directed(X: np.ndarray, source: int, target: int,
                   max_lag: int) -> tuple[float, int, float]:
    T = X.shape[0]
    best_corr, best_lag = 0.0, 0
    for lag in range(1, max_lag + 1):
        if T - lag < 3:
            break
        c = _corr(X[: T - lag, source], X[lag:, target])
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, lag
    return abs(best_corr), best_lag, best_corr


def _normalise_name(value: str) -> str:
    return (value or "").strip().lower()


def _topology_index(prepared: PreparedIncident, keys: list[tuple],
                    topology_nodes: list[TopologyNodeIn],
                    topology_edges: list[TopologyEdgeIn]) -> tuple[set[tuple[int, int]], int]:
    name_to_idx: dict[str, set[int]] = defaultdict(set)
    for idx, key in enumerate(keys):
        for node, alias in prepared.device_sets.get(key, set()):
            for value in (node, alias):
                norm = _normalise_name(value)
                if norm:
                    name_to_idx[norm].add(idx)
        name_to_idx[_normalise_name(series_id(key))].add(idx)
    for node in topology_nodes:
        idxs = set()
        for value in (node.id, node.node, node.node_alias):
            norm = _normalise_name(value)
            if norm:
                idxs |= name_to_idx.get(norm, set())
        for value in (node.node, node.node_alias):
            norm = _normalise_name(value)
            if norm:
                name_to_idx[norm] |= idxs

    related: set[tuple[int, int]] = set()
    used = 0
    for edge in topology_edges:
        left = name_to_idx.get(_normalise_name(edge.source), set())
        right = name_to_idx.get(_normalise_name(edge.target), set())
        if not left or not right:
            continue
        used += 1
        for a in left:
            for b in right:
                if a != b:
                    related.add(tuple(sorted((a, b))))
    return related, used


def _find_cycle(n: int, edges: list[dict]) -> list[int] | None:
    succ: dict[int, list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        succ[edge["source_idx"]].append(idx)
    colour = [0] * n
    stack: list[int] = []
    stack_edge: list[int] = []

    def dfs(node: int) -> list[int] | None:
        colour[node] = 1
        stack.append(node)
        for edge_idx in succ.get(node, []):
            nxt = edges[edge_idx]["target_idx"]
            if colour[nxt] == 0:
                stack_edge.append(edge_idx)
                found = dfs(nxt)
                if found is not None:
                    return found
                stack_edge.pop()
            elif colour[nxt] == 1:
                try:
                    start = stack.index(nxt)
                except ValueError:
                    start = 0
                cycle_nodes = stack[start:] + [nxt]
                cycle_edges = []
                for a, b in zip(cycle_nodes, cycle_nodes[1:]):
                    for idx in succ.get(a, []):
                        if edges[idx]["target_idx"] == b:
                            cycle_edges.append(idx)
                            break
                return cycle_edges
        stack.pop()
        colour[node] = 2
        return None

    for node in range(n):
        if colour[node] == 0:
            found = dfs(node)
            if found:
                return found
    return None


def _remove_cycles(n: int, edges: list[dict]) -> tuple[list[dict], int]:
    removed = 0
    while edges:
        cycle = _find_cycle(n, edges)
        if not cycle:
            break
        weakest = min(cycle, key=lambda idx: edges[idx]["causal_power"])
        edges = [edge for idx, edge in enumerate(edges) if idx != weakest]
        removed += 1
        if removed > n * 2:
            break
    return edges, removed


def _topological_order(n: int, edges: list[dict], node_ids: list[str]) -> list[str]:
    indeg = [0] * n
    succ: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        succ[edge["source_idx"]].append(edge["target_idx"])
        indeg[edge["target_idx"]] += 1
    ready = sorted((i for i in range(n) if indeg[i] == 0), key=lambda i: node_ids[i])
    order = []
    while ready:
        node = ready.pop(0)
        order.append(node_ids[node])
        for nxt in sorted(succ.get(node, []), key=lambda i: node_ids[i]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort(key=lambda i: node_ids[i])
    if len(order) < n:
        order.extend(node_ids[i] for i in range(n) if node_ids[i] not in order)
    return order


def build_causal_graph(prepared: PreparedIncident, config: CausalConfig | None = None,
                       topology_nodes: list[TopologyNodeIn] | None = None,
                       topology_edges: list[TopologyEdgeIn] | None = None) -> dict:
    """Build a lagged, acyclic causal graph from EWMA alarm signals.

    The skeleton is selected with a PC-style conditional-independence test; edges
    are oriented by the strongest lagged direction and cycles are removed. This is
    a deterministic, dependency-light fallback for CausalNex/DoWhy in the hot path.
    """
    config = config or CausalConfig()
    if topology_nodes is None and prepared.bundle is not None:
        topology_nodes = list(prepared.bundle.topology_nodes)
    if topology_edges is None and prepared.bundle is not None:
        topology_edges = list(prepared.bundle.topology_edges)
    topology_nodes = topology_nodes or []
    topology_edges = topology_edges or []

    warnings = list(prepared.warnings)
    window = dict(start=prepared.start, end=prepared.end,
                  step_seconds=prepared.step_seconds,
                  ewma_alpha=prepared.ewma_alpha)
    if prepared.empty:
        return dict(incident_id=prepared.incident_id, nodes=[], edges=[],
                    topological_order=[], method="lagged_pc", window=window,
                    warnings=warnings + ["no series to build a graph"])

    candidates = [
        key for key in prepared.keys
        if prepared.meta[key]["active_bins"] >= max(1, int(config.min_active_bins))
    ]
    candidates.sort(key=lambda k: (-prepared.meta[k]["active_bins"],
                                   -prepared.meta[k]["events"], series_id(k)))
    if len(candidates) > config.max_nodes:
        warnings.append(
            f"causal graph limited to {config.max_nodes} of {len(candidates)} active series")
        candidates = candidates[:config.max_nodes]
    keys = candidates
    n = len(keys)
    node_ids = [series_id(key) for key in keys]
    nodes = [
        dict(
            id=series_id(key),
            label=f"{key[1]}|{key[2]}",
            integration=key[0],
            system=key[1],
            alert_group=key[2],
            events=prepared.meta[key]["events"],
            active_bins=prepared.meta[key]["active_bins"],
            duty=prepared.meta[key]["duty"],
            devices=prepared.meta[key]["devices"],
            regions=prepared.meta[key]["regions"],
            severity_max=prepared.meta[key]["severity_max"],
            peak_intensity=prepared.meta[key]["peak_intensity"],
            peak_ewma=prepared.meta[key]["peak_ewma"],
        )
        for key in keys
    ]
    if n < 2:
        return dict(incident_id=prepared.incident_id, nodes=nodes, edges=[],
                    topological_order=node_ids, method="lagged_pc", window=window,
                    warnings=warnings + ["fewer than two active series"])

    T = len(prepared.grid)
    max_lag_bins = max(1, int(round(config.max_lag_seconds / max(prepared.step_seconds, 1))))
    max_lag_bins = min(max_lag_bins, max(1, (T - 1) // 2))
    if T < 5:
        warnings.append("short window: causal lags may be unstable")

    X = prepared.ewma[node_ids].to_numpy(dtype=float)
    if config.remove_common_factor and n >= 4:
        X = _remove_common_factor(X)
    elif config.remove_common_factor:
        warnings.append("common-factor removal skipped: fewer than 4 series")
    X = _zscore(X)

    related, topo_used = _topology_index(prepared, keys, topology_nodes, topology_edges)
    if config.use_topology and topology_edges:
        if not related:
            warnings.append("topology supplied but no series matched its nodes; constraint skipped")
        elif not config.topology_strict:
            warnings.append(f"topology used as context only ({topo_used} edge(s) matched)")
        else:
            warnings.append(f"strict topology constraint applied ({len(related)} series pair(s))")

    def topology_allowed(a: int, b: int) -> bool:
        if not (config.use_topology and config.topology_strict and related):
            return True
        return tuple(sorted((a, b))) in related

    candidates_edges: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            if not topology_allowed(i, j):
                continue
            forward = _best_directed(X, i, j, max_lag_bins)
            reverse = _best_directed(X, j, i, max_lag_bins)
            if max(forward[0], reverse[0]) < config.min_abs_corr:
                continue
            if forward[0] >= reverse[0]:
                source, target, lag, corr = i, j, forward[1], forward[2]
                power = forward[0]
            else:
                source, target, lag, corr = j, i, reverse[1], reverse[2]
                power = reverse[0]
            candidates_edges.append(dict(source_idx=source, target_idx=target,
                                         lag_bins=lag, corr=corr, causal_power=power,
                                         partial_corr=power))

    edges: list[dict] = []
    for edge in candidates_edges:
        i, j = edge["source_idx"], edge["target_idx"]
        lag = edge["lag_bins"]
        others = [k for k in range(n) if k not in (i, j)]
        if len(others) > config.max_cond_candidates:
            scores = []
            for k in others:
                s = abs(_corr(X[:, i], X[:, k])) + abs(_corr(X[:, j], X[:, k]))
                scores.append((s, k))
            others = [k for _, k in sorted(scores, reverse=True)[:config.max_cond_candidates]]
        min_partial = edge["causal_power"]
        if others and config.max_cond_size > 0:
            for size in range(1, min(config.max_cond_size, len(others)) + 1):
                for cond in itertools.combinations(others, size):
                    partial = abs(_partial_corr_at_lag(X, i, j, lag, list(cond)))
                    min_partial = min(min_partial, partial)
                    if min_partial < config.min_abs_corr:
                        break
                if min_partial < config.min_abs_corr:
                    break
        edge["partial_corr"] = min_partial
        edge["causal_power"] = min_partial
        if min_partial >= config.min_abs_corr and min_partial >= config.min_causal_power:
            edges.append(edge)

    edges.sort(key=lambda e: (-e["causal_power"], e["lag_bins"]))
    if len(edges) > config.max_edges:
        warnings.append(f"causal edge list truncated to {config.max_edges}")
        edges = edges[:config.max_edges]

    edges, removed = _remove_cycles(n, edges)
    if removed:
        warnings.append(f"removed {removed} edge(s) to keep the graph acyclic")

    order = _topological_order(n, edges, node_ids)
    payload_edges = [
        dict(
            **{"from": node_ids[edge["source_idx"]], "to": node_ids[edge["target_idx"]]},
            causal_power=round(float(edge["causal_power"]), 4),
            treatment_effect=round(float(edge["corr"]), 4),
            corr=round(float(edge["corr"]), 4),
            partial_corr=round(float(edge["partial_corr"]), 4),
            lag_bins=int(edge["lag_bins"]),
            lag_seconds=int(edge["lag_bins"] * prepared.step_seconds),
            method="lagged_pc",
        )
        for edge in edges
    ]
    if not payload_edges:
        warnings.append("no causal edges passed the thresholds; collect a longer window or lower min_abs_corr")
    return dict(
        incident_id=prepared.incident_id,
        nodes=nodes,
        edges=payload_edges,
        topological_order=order,
        method="lagged_pc",
        window=window,
        topology=dict(nodes=len(topology_nodes), edges=len(topology_edges),
                      matched_edges=topo_used, strict=bool(config.topology_strict)),
        warnings=warnings,
    )
