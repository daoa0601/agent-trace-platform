"""Collector metrics isolated per application instance for reliable tests."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class CollectorMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "agenttrace_ingest_requests_total",
            "Ingestion requests by protocol and outcome.",
            ["protocol", "outcome"],
            registry=self.registry,
        )
        self.events = Counter(
            "agenttrace_ingested_events_total",
            "Validated events accepted by protocol.",
            ["protocol"],
            registry=self.registry,
        )
        self.duplicates = Counter(
            "agenttrace_duplicate_events_total",
            "Events in byte-identical batches already present in raw storage.",
            ["protocol"],
            registry=self.registry,
        )
        self.redactions = Counter(
            "agenttrace_redactions_total",
            "Collection-boundary redactions by category.",
            ["category"],
            registry=self.registry,
        )
        self.raw_bytes = Counter(
            "agenttrace_raw_bytes_total",
            "Compressed raw bytes persisted.",
            registry=self.registry,
        )
        self.write_seconds = Histogram(
            "agenttrace_raw_write_seconds",
            "Raw batch persistence latency.",
            ["protocol"],
            registry=self.registry,
        )
        self.ready = Gauge(
            "agenttrace_collector_ready",
            "Whether the object store passed the most recent readiness check.",
            registry=self.registry,
        )

    def observe_redactions(self, stats: object) -> None:
        for category in ("secret_values", "secret_keys", "pii_values", "private_reasoning"):
            count = int(getattr(stats, category, 0))
            if count:
                self.redactions.labels(category=category).inc(count)
