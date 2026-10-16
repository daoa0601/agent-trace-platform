"""Immutable audit-bundle persistence on the AgentTrace object store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from agenttrace.audits import persist_audit_bundle
from agenttrace.storage import FileObjectStore
from benchmark_audit_core import (
    AuditableSession,
    AuditorKind,
    AuditVerdict,
    BenchmarkClaim,
    BenchmarkStage,
    EmpiricalRetest,
    EvidenceRef,
    GateStatus,
    HardwareProfile,
    IndependentAudit,
    OptimizationScope,
    RetestCategory,
    RetestOutcome,
    TerminationReason,
    build_audit_bundle,
)


def _evidence(name: str) -> EvidenceRef:
    digest = {"trace": "1", "artifact": "2", "hardware": "3", "claim": "4"}.get(name, "5")
    return EvidenceRef(
        uri=f"object://evidence/{name}",
        sha256=digest * 64,
        media_type="application/json",
        byte_size=100,
    )


def _audit(ended_at: datetime) -> IndependentAudit:
    retests = [
        EmpiricalRetest(
            retest_id=f"retest-{category.value}",
            category=category,
            outcome=RetestOutcome.PASSED,
            isolated_from_original_session=True,
            evidence=[_evidence(f"retest-{category.value}")],
        )
        for category in (
            RetestCategory.FRESH_PROCESS,
            RetestCategory.COLD_CACHE,
            RetestCategory.INPUT_PERTURBATION,
            RetestCategory.OUTPUT_VALIDATION,
        )
    ]
    return IndependentAudit(
        audit_id="audit-001",
        auditor_id="review-agent-02",
        auditor_kind=AuditorKind.HYBRID,
        independent=True,
        manual_review_completed=True,
        reviewed_final_artifact=True,
        reviewed_full_trace=True,
        reviewed_task_definition=True,
        reviewed_benchmark_protocol=True,
        retests=retests,
        verdict=AuditVerdict.PASS,
        completed_at=ended_at + timedelta(minutes=30),
    )


def _session(*, headline: bool = True, source_dirty: bool = False) -> AuditableSession:
    started_at = datetime(2026, 7, 17, 8, 0, tzinfo=UTC)
    ended_at = started_at + timedelta(hours=2)
    return AuditableSession(
        session_id="session-001",
        experiment_id="kernel-study-v1",
        run_id="run-001",
        episode_id="episode-001",
        trace_id="a" * 32,
        task_id="systems.kernel-optimization",
        task_version="1.0.0",
        generator_id="optimizer-agent-01",
        source_commit="abcdef1234567890",
        source_dirty=source_dirty,
        benchmark_protocol_version="kernel-v1",
        task_definition=_evidence("task-definition"),
        benchmark_protocol=_evidence("benchmark-protocol"),
        started_at=started_at,
        ended_at=ended_at,
        termination_reason=TerminationReason.AGENT_FINISHED,
        hardware=HardwareProfile(
            accelerator_vendor="NVIDIA",
            accelerator_models=["H100"],
            accelerator_count=1,
            optimization_scope=OptimizationScope.SINGLE_DEVICE,
            host_architecture="x86_64",
            driver_version="test-driver",
            runtime_versions={"cuda": "test-runtime"},
            observed_evidence=_evidence("hardware"),
        ),
        stage_counts={
            BenchmarkStage.COMPILE: 4,
            BenchmarkStage.CHECK: 4,
            BenchmarkStage.BENCHMARK: 12,
        },
        benchmark_repetitions=10,
        full_trace=_evidence("trace"),
        final_artifact=_evidence("artifact"),
        claims=[
            BenchmarkClaim(
                claim_id="latency-main",
                metric_name="median latency",
                value=12.5,
                unit="microseconds",
                headline=headline,
                evidence=[_evidence("claim")],
            )
        ],
        audit=_audit(ended_at),
    )


def test_persist_audit_bundle_is_content_addressed_and_never_overwrites(tmp_path: Path) -> None:
    session = _session()
    first = build_audit_bundle(session)
    second = build_audit_bundle(session)
    assert first.headline_gate.status == GateStatus.PASSED

    store = FileObjectStore(tmp_path)
    created = persist_audit_bundle(store, first)
    duplicate = persist_audit_bundle(store, second)

    assert created.key == f"audits/bundles/{first.digest()}.json"
    assert created.created is True
    assert duplicate.created is False
    assert duplicate.key == created.key
    assert store.get_bytes(created.key) == first.canonical_bytes()
