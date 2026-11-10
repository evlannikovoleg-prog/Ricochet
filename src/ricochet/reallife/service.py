from __future__ import annotations

from dataclasses import dataclass

from .causal import CausalConfig, build_causal_graph
from .clustering import ClusterConfig, cluster_series
from .pipeline import PreparedIncident, prepare_incident
from .runbook import RunbookAgent, RunbookConfig
from .schemas import IncidentBundleIn


@dataclass
class RealtimeOptions:
    step_seconds: int = 10
    ewma_half_life_seconds: float = 60.0
    max_bins: int = 3000
    max_series: int = 5000


class RealtimeService:
    """One preprocessing pipeline for all three realtime endpoints."""

    def __init__(self, options: RealtimeOptions | None = None,
                 runbook_agent: RunbookAgent | None = None):
        self.options = options or RealtimeOptions()
        self.runbook_agent = runbook_agent or RunbookAgent()

    def prepare(self, bundle: IncidentBundleIn) -> PreparedIncident:
        return prepare_incident(
            bundle,
            step_seconds=self.options.step_seconds,
            ewma_half_life_seconds=self.options.ewma_half_life_seconds,
            max_bins=self.options.max_bins,
            max_series=self.options.max_series,
        )

    def causal_graph(self, bundle: IncidentBundleIn,
                     config: CausalConfig | None = None) -> dict:
        prepared = self.prepare(bundle)
        return build_causal_graph(prepared, config)

    def clusters(self, bundle: IncidentBundleIn,
                 config: ClusterConfig | None = None) -> dict:
        prepared = self.prepare(bundle)
        return cluster_series(prepared, config)

    def runbook(self, bundle: IncidentBundleIn, *, causal_config: CausalConfig | None = None,
                cluster_config: ClusterConfig | None = None,
                runbook_config: RunbookConfig | None = None,
                ranker=None, runbooks=None, incidents=None) -> dict:
        prepared = self.prepare(bundle)
        causal = build_causal_graph(prepared, causal_config)
        clusters = cluster_series(prepared, cluster_config)
        return self.runbook_agent.generate(
            prepared, causal, clusters,
            ranker=ranker, runbooks=runbooks, incidents=incidents,
            config=runbook_config,
        )
